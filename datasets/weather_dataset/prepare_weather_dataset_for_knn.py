"""
Script used to prepare the weather dataset to be able to use it with the kNN classifier.
The script reads in a certain number of rows of the csv file (the csv file is too big to
read the whole file at once). Also only the relevant numerical columns are read in and
the column with the class of the sample (name of the weather station).
From these read in rows, then a certain number of rows are sampled and stored in a new 
csv file.
"""

import random
import pandas as pd

# PARAMETERS:
NUMBER_OF_ROWS_TO_SAMPLE_FROM = 1000000
NUMBER_OF_ROWS_TO_SAMPLE = 10000 # must be smaller or equal to NUMBER_OF_ROWS_TO_SAMPLE_FROM
NUMBER_OF_COLUMNS = 8 # all numerical columns = 18
INPUT_FILE = "weather_data_south.csv"
OUTPUT_FILE = "weather_data_south_prepared.csv"

# Numerical column names + class column name ("station")
relevant_column_names = ["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)","PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)", "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)", "RADIACAO GLOBAL (Kj/m²)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "TEMPERATURA DO PONTO DE ORVALHO (°C)", "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)", "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)", "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)", "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)", "VENTO, DIREÇÃO HORARIA (gr) (° (gr))", "VENTO, RAJADA MAXIMA (m/s)",
    "VENTO, VELOCIDADE HORARIA (m/s)", "height", "station"]

# sample NUMBER_OF_COLUMNS columns
relevant_column_names[:-1] = random.sample(relevant_column_names[:-1], NUMBER_OF_COLUMNS)

# read in NUMBER_OF_COLUMNS columns and NUMBER_OF_ROWS_TO_SAMPLE_FROM rows from csv file
df = pd.read_csv(INPUT_FILE, header=0, nrows=NUMBER_OF_ROWS_TO_SAMPLE_FROM, usecols=relevant_column_names)

# Sample random columns to decrease size of csv (random_state is the seed)
df_sampled = df.sample(NUMBER_OF_ROWS_TO_SAMPLE, random_state=1)

# Replace whitespaces with "_" so that file reading is easier in C
df_sampled.replace(" ", "_", regex=True, inplace=True)

# Save sampled csv
df_sampled.to_csv(OUTPUT_FILE, header=False, index=False)