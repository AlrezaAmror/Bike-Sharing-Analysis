# Submission Dicoding "Belajar Analisis Data dengan Python"<br>
This is a final project from Dicoding in the "Belajar Analisis Data Dengan Python" course to make analysis. 
In the notebook file, I attached the way I did the analysis from Data Wrangling, Exploratory Data Analysis, and Data Visualization. 
Besides that, I also make the dashboard using streamlit

## Preview
![Dashboard](https://raw.githubusercontent.com/AlrezaAmror/Bike-Sharing-Analysis/main/preview1.png)

## File Structures
```
.
├── dashboard
│   ├── dashboard.py
│   ├── day.csv
│   ├── hour.csv
│   └── logo.png
├── data
│   ├── day.csv
|   └── hour.csv
├── preview1.png
├── README.md
├── notebook.ipynb
├── url.txt
└── requirements.txt
```

## Project work flow
1. Data Wrangling: 
 - Gathering data
 - Assessing data
 - Cleaning data
2. Exploratory Data Analysis:
 - Defined business questions for data exploration
 - Create Data exploration
3. Data Visualization:
 - Create Data Visualization that answer business questions
4. Dashboard:
 - Make filter components on the dashboard
 - Complete the dashboard with various data visualizations

## Instalation

1. Clone repository

   ```shell
   git clone https://github.com/AlrezaAmror/Bike-Sharing-Analysis.git
   ```

2. Install requirement

    ```shell
    pip install streamlit
    pip install -r requirements.txt
    ```

## Use
1. Go to your local directory:

    ```shell
    cd bike-sharing/dashboard/
    streamlit run dashboard.py
