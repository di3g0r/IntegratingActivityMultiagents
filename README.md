# Multiagent Systems Integrating Activity

## Participants
- **Diego Alejandro Calvario Aceves A01642806**  
- **Diego Rodríguez Romero A01741413**  
- **Milan Alejandro De Alba Gómez A01637667**  
- **Gonzalo Calderón Gil Leyva A01740008**  

---

## Overview  
This project simulates a multi-agent system within a warehouse environment. Autonomous robot agents collaborate to organize objects into designated stacks efficiently. Robots navigate a grid, detect objects, avoid collisions, and make decisions to optimize the organization process.

---

## Agent Specifications  
Each robot agent has:  
- **Properties**: Position, direction, unique ID, carrying status, and target stack.  
- **Behaviors**: Movement, object detection, collision avoidance, and object pickup.  

Agents operate using a **hybrid architecture**:  
- **Reactive Layer**: Handles real-time interactions (e.g., obstacle avoidance, immediate object handling).  
- **Deliberative Layer**: Manages long-term goals like selecting stacks and organizing objects.  

---

## Environment Specifications  
The warehouse environment is a grid with:  
- **Cells**: Representing objects, empty spaces, or stacks (up to 5 objects high).  
- **Obstacles**: Randomly placed at the simulation start.  
- **Ontology Integration**: Formalizes robot actions and environmental states for dynamic interaction.  

---

## Success Metrics  
Efficiency is measured by the **average steps taken per object placement**. Lower averages indicate better organization and pathfinding by the robots.  

---

This simulation explores the synergy between reactive and deliberative strategies in multi-agent systems, emphasizing efficiency and adaptability in dynamic environments.

---

## Links to Video and Unity Project  
- **Video**: [Link](https://drive.google.com/file/d/1yfpKzJ3jPrWxiQpsySeExBUQF62C_zJy/view?usp=drive_link)  
- **Unity Project**: [Link](https://drive.google.com/file/d/1CHX_GwcNPtOmdBkGy0Y2Qs3uihZLXe9b/view?usp=sharing)  

