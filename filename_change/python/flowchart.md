# Flowchart

```mermaid
graph TD
    A[Start] --> B{Enter Base Name};
    B --> C{Select Property};
    C --> D{Add File(s)};
    D --> E{More than one file?};
    E -- No --> F[Rename File];
    E -- Yes --> G{Are all files images?};
    G -- Yes --> H{Select Output Format};
    H --> I[Merge Files];
    I --> F;
    G -- No --> D;
    F --> J{Another file?};
    J -- Yes --> C;
    J -- No --> K[End];
```

## Flowchart Description

1.  **Start:** The program starts.
2.  **Enter Base Name:** The user is prompted to enter the base name for the files.
3.  **Select Property:** The user selects a property from the list (e.g., "영수증").
4.  **Add File(s):** The user adds one or more files for the selected property.
5.  **More than one file?:** The program checks if the user added more than one file.
6.  **Rename File:** If only one file was added, the program renames it to the format `{base_name}_{property}.{extension}`.
7.  **Are all files images?:** If more than one file was added, the program checks if all the files are images.
8.  **Select Output Format:** If all the files are images, the user is prompted to select an output format for the merged file (e.g., PDF).
9.  **Merge Files:** The program merges the image files into a single file with the specified format.
10. **Another file?:** The program asks the user if they want to rename another file.
11. **End:** The program ends.
