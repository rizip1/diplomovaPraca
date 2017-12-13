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
All options can be set in `conf.py` file:
```
python main.py
```

# Data analysis script
Run some data analysis tasks:
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
