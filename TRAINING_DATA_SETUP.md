# Training Data Setup Guide

This project has been refactored to separate training data generation from the main application.

## Workflow

### 1. Generate Training Data (First Time Setup)

Run the Jupyter notebook to generate the training data CSV file:

```bash
# Open the notebook
jupyter notebook build_training_data.ipynb
```

**Or** open [build_training_data.ipynb](build_training_data.ipynb) in VS Code and run all cells.

This will:
- Generate 15,000 rows of physics-based training data
- Create visualizations of the data distribution
- Export the data to `training_data.csv`
- Verify the saved file

### 2. Run the Main Application

Once `training_data.csv` exists, you can run the main Dash app:

```bash
python app.py
```

The app will automatically load the training data from the CSV file and train the models.

## Files

- **build_training_data.ipynb**: Notebook that generates training data
- **training_data.csv**: Generated training data (15,000 rows, 6 columns)
- **app.py**: Main Dash application that loads and uses the training data

## Benefits of This Approach

1. **Separation of Concerns**: Data generation is separate from model training/deployment
2. **Reproducibility**: Training data is generated once and stored in CSV format
3. **Inspection**: You can inspect and visualize the training data before using it
4. **Performance**: Faster app startup since data generation happens offline
5. **Version Control**: Training data can be versioned and tracked

## Data Schema

The `training_data.csv` file contains the following columns:

- **Outdoor_Temp**: Outdoor temperature (°C)
- **Prev_Indoor_Temp**: Previous indoor temperature (°C)
- **Setpoint**: HVAC setpoint temperature (°C)
- **Occupancy**: Room occupancy status (0 = empty, 1 = occupied)
- **Target_Temp**: Target next indoor temperature (°C)
- **Target_Energy**: Target energy consumption (kWh)

## Troubleshooting

If you get an error about missing `training_data.csv`:
1. Make sure you've run the `build_training_data.ipynb` notebook
2. Verify that `training_data.csv` exists in the project root directory
3. Check that the CSV file contains all required columns
