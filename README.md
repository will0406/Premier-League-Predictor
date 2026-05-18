# Premier-League-Predictor
LLM-Powered Football Match Predictor and Analysis Assistant 

This project is a match forecasting system that combines mathematical Poisson-Dixon-Coles modeling with Ollama (via RAG) to provide data-driven match predictions and analyst-style explanations.

Prerequisites:
1. Install Ollama from Ollama.com, run: ollama run llama3.2.
2. Install Python packages: pip install pandas numpy scipy ollama.
3. Have data-set.csv and predictor 2.0.py in the same folder.

How to Run:
1. Open project in VS Code or any other editor, make sure data-set and predictor are in same directory.
2. Run code.
3. Input home and away team, the terminal will show which teams you can select.
   Some team name mappings:
    "man u": "Man United",
    "man utd": "Man United",
    "manchester united": "Man United",
    "man city": "Man City",
    "manchester city": "Man City",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "wolves": "Wolves",
    "forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "brighton and hove albion": "Brighton",
    "leicester city": "Leicester",
    "west ham united": "West Ham",
    "newcastle united": "Newcastle",
    "sheffield utd": "Sheffield United",
    "leeds united": "Leeds"

About Dataset:
Date: the date the match was played, DD/MM/YYYY.
Season: which season the match belonged to.
HomeTeam: the team to play at home.
AwayTeam: the team to play away.
FTHG: Full Time Home Goal, number of goals home team scored at full time.
FTAG: Full Time Away Goal, number of goals away team scored at full time.
FTR: Full Time Result, the result of the match, H meaning home win, A meaning away win, and D meaning draw.
HST: Home Shots on Target, the amount of shots on target for the home team.
AST: Away Shots on Target, the amount of shots on target for the away team.
HY: Home Yellow, amount of yellow cards for the home team.
AY: Away Yellow, amount of yellow cards for the away team.
AvgH: the average market odds for the home team to win.
AvgD: the average market odds for a draw.
AvgA: the average market odds for the away team to win.
