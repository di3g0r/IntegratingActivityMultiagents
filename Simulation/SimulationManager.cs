using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class SimulationManager : MonoBehaviour
{
    [Header("Flask API Configuration")]
    public string apiUrl = "http://127.0.0.1:5000";

    [Header("Simulation Configuration")]
    public int width = 15;
    public int height = 15;
    public int numRobots = 5;
    public int numObjects = 20;

    [Header("Visualization")]
    public GameObject robotPrefab;
    public GameObject objectPrefab;
    public GameObject stackPrefab;
    public GameObject wallPrefab;
    
    [Header("Movement Configuration")]
    public float movementSpeed = 2f;
    public float rotationSpeed = 360f;
    public float objectStackHeight = 0.5f;
    
    private Dictionary<int, GameObject> robotObjects = new Dictionary<int, GameObject>();
    private GameObject[,] gridObjects;
    private Dictionary<Vector2Int, List<GameObject>> stackObjects = new Dictionary<Vector2Int, List<GameObject>>();

    void Start()
    {
        Debug.Log("Starting simulation...");
        gridObjects = new GameObject[width, height];
        StartCoroutine(InitializeSimulation());
    }

    IEnumerator InitializeSimulation()
    {
        Debug.Log("Initializing simulation...");
        string initializeUrl = $"{apiUrl}/initialize";
        string jsonBody = JsonUtility.ToJson(new
        {
            width = this.width,
            height = this.height,
            num_robots = this.numRobots,
            num_objects = this.numObjects
        });

        UnityWebRequest request = new UnityWebRequest(initializeUrl, "POST");
        byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(jsonToSend);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Failed to initialize simulation: {request.error}");
        }
        else
        {
            Debug.Log("Simulation initialized successfully!");
            // Create initial walls
            CreateWalls();
            StartCoroutine(RunSimulation());
        }
    }

    private void CreateWalls()
    {
        // Create walls around the perimeter
        for (int x = 0; x < width; x++)
        {
            CreateWall(x, 0);  // Bottom wall
            CreateWall(x, height - 1);  // Top wall
        }
        for (int y = 0; y < height; y++)
        {
            CreateWall(0, y);  // Left wall
            CreateWall(width - 1, y);  // Right wall
        }
    }

    private void CreateWall(int x, int y)
    {
        GameObject wall = Instantiate(wallPrefab, new Vector3(x, 0.5f, y), Quaternion.identity);
        gridObjects[x, y] = wall;
    }

    IEnumerator RunSimulation()
    {
        Debug.Log("Starting simulation loop...");
        while (true)
        {
            yield return StartCoroutine(StepSimulation());
            yield return StartCoroutine(GetSimulationState());
            yield return new WaitForSeconds(0.5f);
        }
    }

    IEnumerator StepSimulation()
    {
        string stepUrl = $"{apiUrl}/step";
        UnityWebRequest request = new UnityWebRequest(stepUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(new byte[0]);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Step simulation error: {request.error}");
        }
    }

     IEnumerator GetSimulationState()
    {
        string stateUrl = $"{apiUrl}/simulation-state";
        UnityWebRequest request = UnityWebRequest.Get(stateUrl);
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Error fetching simulation state: {request.error}");
            yield break;
        }

        string jsonResponse = request.downloadHandler.text;
        
        try
        {
            SimulationState state = JsonConvert.DeserializeObject<SimulationState>(jsonResponse);
            if (state == null || state.grid == null)
            {
                Debug.LogError("Deserialization failed: state or grid is null");
                yield break;
            }

            // Log robot states
            foreach (var robot in state.robots)
            {
                if (robot.carrying_object)
                {
                    Debug.Log($"Robot {robot.id} at ({robot.x}, {robot.y}) is carrying an object. State: {robot.state}");
                }
            }

            // Log stack information
            foreach (var stack in state.stacks)
            {
                Debug.Log($"Stack at ({stack.x}, {stack.y}) has {stack.height} objects");
            }

            int[,] gridArray = ConvertTo2DArray(state.grid);
            UpdateVisualization(gridArray, state.robots, state.stacks);
        }
        catch (JsonException e)
        {
            Debug.LogError($"JSON Deserialization error: {e.Message}");
        }
    }

    void UpdateVisualization(int[,] grid, List<RobotState> robots, List<StackState> stacks)
    {
        // Update objects
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (gridObjects[x, y] != null && grid[y, x] == (int)CellType.EMPTY)
                {
                    // Only destroy if it's not a wall
                    if (grid[y, x] != (int)CellType.WALL)
                    {
                        Destroy(gridObjects[x, y]);
                        gridObjects[x, y] = null;
                    }
                }
                else if (grid[y, x] == (int)CellType.OBJECT && gridObjects[x, y] == null)
                {
                    GameObject obj = Instantiate(objectPrefab, new Vector3(x, 0.5f, y), Quaternion.identity);
                    gridObjects[x, y] = obj;
                }
            }
        }

        // Update robots
        foreach (var robot in robots)
        {
            GameObject robotObj;
            Vector3 position = new Vector3(robot.x, 1f, robot.y);
            Quaternion rotation = GetRotationFromDirection(robot.direction);

            if (!robotObjects.TryGetValue(robot.id, out robotObj))
            {
                robotObj = Instantiate(robotPrefab, position, rotation);
                robotObjects[robot.id] = robotObj;
            }
            else
            {
                robotObj.transform.position = position;
                robotObj.transform.rotation = rotation;
            }

            // Update robot appearance
            Renderer renderer = robotObj.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = robot.carrying_object ? Color.blue : Color.white;
            }
        }

        // Update stacks
        foreach (var stack in stacks)
        {
            Vector2Int stackPos = new Vector2Int(stack.x, stack.y);
            if (!stackObjects.ContainsKey(stackPos))
            {
                stackObjects[stackPos] = new List<GameObject>();
            }

            // Create or update stack visualization
            while (stackObjects[stackPos].Count < stack.height)
            {
                float yPosition = stackObjects[stackPos].Count * objectStackHeight;
                GameObject stackObj = Instantiate(stackPrefab, 
                    new Vector3(stack.x, yPosition + objectStackHeight/2, stack.y), 
                    Quaternion.identity);
                stackObjects[stackPos].Add(stackObj);
            }
        }
    }

    private Quaternion GetRotationFromDirection(string direction)
    {
        switch (direction)
        {
            case "NORTH":
                return Quaternion.Euler(0, 0, 0);
            case "SOUTH":


                return Quaternion.Euler(0, 180, 0);
            case "EAST":
                return Quaternion.Euler(0, 90, 0);
            case "WEST":
                return Quaternion.Euler(0, -90, 0);
            default:
                return Quaternion.identity;
        }
    }

    private int[,] ConvertTo2DArray(List<List<int>> gridList)
    {
        int rows = gridList.Count;
        int cols = gridList[0].Count;
        int[,] gridArray = new int[rows, cols];
        
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                gridArray[i, j] = gridList[i][j];
            }
        }
        return gridArray;
    }

    [System.Serializable]
    public class SimulationState
    {
        public List<List<int>> grid;
        public List<RobotState> robots;
        public List<StackState> stacks;
    }

    [System.Serializable]
    public class RobotState
    {
        public int id;
        public int x;
        public int y;
        public string direction;
        public bool carrying_object;
        public string state;
    }

    [System.Serializable]
    public class StackState
    {
        public int x;
        public int y;
        public int height;
    }

    public enum CellType
    {
        EMPTY = 0,
        WALL = 1,
        OBJECT = 2,
        ROBOT_EMPTY = 3,
        ROBOT_CARRYING = 4,
        STACK = 5
    }
}