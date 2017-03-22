# Dependencies
python3,
numpy,
pandas,
sklearn,
matplotlib,
psycopg2 (for postgres database connection)

# Prediction script
Example:
```
python predict.py --file data.csv --mode window --weight 0.97 --length 200 --lags 1 --model svr
```
To get info about all possible switches type:
```
python predict.py --help
```

# Data analysis script
Run all data analysis tasks:
```
python data_analysis.py
```
You can choose concreate tasks to skip by specifying appropriate switches.
To lists all possible switches and their descriptions type:
```
python data_analysis.py --help
```
Example:
```
python data_analysis.py --skip-data --skip-invalid
```
