# Dependencies
python3,
numpy,
pandas,
scikit-learn,
matplotlib,
scipy,
keras,
psycopg2 (for postgres database connection)
...

To install dependencies you can create virtualenv in root folder and run
```
pip install -r requirements.txt
```

# Prediction script
Example:
```
python predict.py --file data.csv --model reg --weight 0.97 --length 60 --step 24
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
You can choose concreate tasks by specifying appropriate switches.
To lists all possible switches and their descriptions type:
```
python data_analysis.py --help
```
Example:
```
python data_analysis.py --data --invalid
```
