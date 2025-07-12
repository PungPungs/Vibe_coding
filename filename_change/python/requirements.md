# Requirements Specification

## 1. Introduction

This document outlines the requirements for a file renaming program. The program will allow users to rename files to a standardized format based on a base name and a file type.

## 2. Functional Requirements

### 2.1. File Renaming

- The user shall be able to provide a base name for the file (e.g., "20240524_McDonald").
- The user shall be able to select a file type from a predefined list of properties.
- The program shall rename the selected file to the format: `{base_name}_{property}.{extension}`.

### 2.2. Properties

- The program shall provide a predefined list of properties:
    - 영수증 (Receipt)
    - 거래명세서 (Transaction Statement)
    - 전자세금계산서 (Electronic Tax Invoice)
    - 전표 (Slip)
- The user shall be able to select one of these properties for each file.

### 2.3. File Handling

- The user shall be able to add files one by one.
- The program shall allow the user to associate a file with a property.
- If multiple image files are added for the same property, the program shall merge them into a single file.
- The user shall be able to specify the output format for the merged file (e.g., PDF).

### 2.4. User Interface

- The program shall have a graphical user interface (GUI).
- The GUI shall provide an easy way to enter the base name, select files, and choose properties.

## 3. Non-Functional Requirements

- The program shall be written in Python.
- The program shall be easy to use and intuitive.
- The program shall be able to handle common image formats (e.g., JPG, PNG).
