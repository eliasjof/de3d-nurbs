# DE3NURBS

Welcome to the DE3D-NURBS repository. 
This project is developed by Elias Freitas



## Table of Contents

- [DE3NURBS](#de3nurbs)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Cite us](#cite-us)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Examples](#examples)
  - [Folder Structure](#folder-structure)

## Introduction

DE3D-NURBS is a Python-based tool for a novel path planner, considering kinematics constraints, such as the maximum and minimum climb/dive angle and the maximum curvature imposed by an aerial robot.
For more details, see the reference paper.

### Cite us


## Features

- Generate paths represented by NURBS curves in 3D space
- Novel LSHADE-COP algorithm
  
## Installation

To install DE3D-NURBS, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/eliasjof/de3dnurbs.git
    ```
2. Navigate to the project directory:
    ```sh
    cd de3dnurbs
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use DE3DNURBS, follow these steps:

1. Import the `de3dnurbs` module in your Python script:
    ```python
    import scripts.de3nurbs
    ```


### Examples

The [`examples`](examples/) folder contains sample scripts demonstrating how to use the DE3NURBS library, evaluating our path planner.

To run an example script, navigate to the [`examples`](examples/) folder and execute the script using Python:

```sh
python examples/run_algorithms.py 
```

If SAVE=1 (in [`examples/__experiment_setup.py`](examples/__experiment_setup.py)), the results of the path planning will be saved in the folder [`results/`](results/).


## Folder Structure

- [`examples/run_algorithms.py`](examples/run_algorithms.py): A complete script to run six optimization algorithms with our planner.
- [`scripts/`](scripts/): This folder contains the code used for our planner and the LSHADE-COP algorithm.
- [`results/`](results/): This folder contains the results obtained by the path-planner


