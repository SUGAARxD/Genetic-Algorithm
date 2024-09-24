# Genetic-Algorithm

## Description

This project implements a genetic algorithm designed to optimize the neural network configuration used in [WineQuality-LinearRegression](https://github.com/SUGAARxD/WineQuality-LinearRegression). It searches for the optimal combination of hidden layer neurons, learning rate, and weight decay  to improve model performance during training.

## Technologies

The genetic algorithm uses `PyTorch` to build and train the model.

## Architecture Details

### Genetic Algorithm

- **Codification:** To optimize **three parameters** for configuration, I choose chromosomes that contain a number of genes **divisible by 3 and not less than 6**. Each gene is either 0 or 1, randomly chosen.
  
- **Selection Step:** The algorithm used for selection is `Roulette Selection`. The steps are as follows:
  - Calculate the **fitness** of each individual.
  - Compute the **total fitness** of the population:
    
    ```math
    \text{Total fitness} = \sum_{i=1}^{N} f\left(i\right)
    ```

    ```math
    \text{Where } f\left(i\right) \text{ is the fitness value of individual } i \text{, and } N \text{ is the number of individuals.}
    ```
     
  - Determine the `selection probability` for each individual:
    
    ```math
    P(i) = \frac{f\left(i\right)}{\text{Total fitness}}
    ```

  - Determine the `cumulative probability` for each individual:
    
    ```math
    C(i) = \sum_{j=1}^{i} P(j)
    ```
  - Generate a **random number** between 0 and 1. The selected individual is the first individual whose cumulative probability is bigger than the random number. That individual will be added to a `new population`.
    
  - Repeat the process untill the `new population` is the same size as the `old population`.

- **Crossover Step:** For each individual from the new population, a **random number** is generated between 0 and 1. If the number is **smaller** than the `crossover probability`, the individual will be selected for crossover. If there is an **odd number** of individuals selected, **the last one will be eliminated**.
  
  For each **two individuals selected for crossover**, a **random integer** is choosen between `chromosome length / 6` and `chromosome length - (chromosome length / 6)` to ensure that at least **half of the first or third parameter is retained**.

  The genes of the individuals are **split in half**, and the **new offspring** will be as follows:
  - The ``first child`` will have the first half of the genes from the first parent and the second half from the second parent.
  - The ``second child`` will have the first half of the genes from the second parent and the second half from the first parent.

  The **new offspring will replace the parents** in the population.
  
- **Mutation Step:** For each individual in the population, for **each gene**, a **random number** is generated between 0 and 1. If the number is smaller than the `mutation probability`, the gene mutates, meaning it becomes **0 if it was 1**, and **1 if it was 0**.

### PyTorch Model

- The model consists of **3 layers of neurons**. The first layer is the input layer that has 11 neurons for the 11 features of a wine, the hidden layer has 20 neurons, but it can be modified, and the output layer has one neuron that represents the predicted score. 

- **Activation Functions:** The model uses `Leaky ReLU` in the hidded layer and linear function(no activation) on the output.

- **Optimizer and Loss Function:** For the numpy implementation I use `Gradient Descent` and for the pytorch implementation I use the `Adam` optimizer. Both use `Mean Squared Error (MSE)` as loss function.

- **Regularization:** I use `weight decay` as a form of regularization.

- **Evaluation Metrics:** The model's performance is evaluated using `loss` on the val set.

## Dataset

To optimize the configuration of a neural network, a suitable dataset is essential.

The dataset used for training is [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality). It contains 4,898 instances of wine, each with 11 physicochemical features (e.g., alcohol, acidity) and a quality score ranging from 0 to 10.

## How to Use

### Installation

1. **Clone the repository**
2. **Create and Activate a Virtual Environment Using Conda:**
   ```bash
   conda create --name env_name python=3.10
   ```
   ```bash
   conda activate env_name
   ```
   Replace `env_name` with your desired environment name.
   
4. **Install the Packages and Dependencies:**

   **Using Conda:**
   ```bash
   conda install tensorboard numpy=1.26.4 pandas pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

   **Or Using Pip:**

   ```bash
   pip install tensorboard numpy=1.26.4 pandas torch torchvision
   ```
### Running the code

You can use a code editor like Visual Studio Code or an IDE like PyCharm. Use your environment with all packages installed.

Alternatively, you can run the script from the console:

1. **Enter the Project Folder:**
   For the numpy version:
   ```bash
   cd path_to_repo\Genetic-Algorithm
   ```
   For the pytorch version:
   ```bash
   cd path_to_repo\Genetic-Algorithm
   ```
   **Make sure to replace `path_to_repo` with the actual path to your cloned repository.**
3. **Run the Code:**
   ```bash
   python main.py
   ```
   
   **Or**

   ```bash
   python3 main.py
   ```
   If the first command does not work.
