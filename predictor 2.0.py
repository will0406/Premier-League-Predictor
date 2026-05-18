import pandas as pd
import numpy as np
from scipy.stats import poisson
import ollama 

# ==========================================
# 1. TEAM NAME MAPPING
# ==========================================
TEAM_NAME_MAPPING = {
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
}

def standardize_team_name(user_input, valid_teams):
    search_term = user_input.strip().lower()
    if search_term in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[search_term]
    for team in valid_teams:
        if search_term == team.lower():
            return team
    return None

# ==========================================
# 2. LLM INTEGRATION (Ollama)
# ==========================================
def get_llm_explanation(home, away, score, h2h_text, h_stats, a_stats, odds):
    h_sot = h_stats['avg_sot']
    a_sot = a_stats['avg_sot']
    
    if h_sot > a_sot:
        sot_leader = f"{home} has the edge"
    elif a_sot > h_sot:
        sot_leader = f"{away} has the edge"
    else:
        sot_leader = "Both teams are equal in attacking pressure"

    h_cards = h_stats['avg_c']
    a_cards = a_stats['avg_c']
    discipline_note = ""
    if h_cards > 2.0: discipline_note = f"{home} averages high yellow cards ({h_cards:.1f})."
    if a_cards > 2.0: discipline_note += f" {away} averages high yellow cards ({a_cards:.1f})."

    prompt = f"""
    Be a professional football data analyst. 
    Explain this match prediction: {home} vs {away}.
    
    Data:
    - Predicted Score: {home} {score[0]} - {score[1]} {away}
    - Market Odds: {odds}
    - {home}: {h_sot:.1f} Avg SOT, {h_cards:.1f} Avg Yellow Cards
    - {away}: {a_sot:.1f} Avg SOT, {a_cards:.1f} Avg Yellow Cards
    - Statistical Advantage: {sot_leader} has a higher SOT average.
    - Discipline Note: {discipline_note if discipline_note else "Both teams showing standard discipline."}

    Provide a concise, 2-sentence tactical explanation. 
    Compare the Poisson score with the SOT advantage and mention if the yellow card averages suggest a risk of bookings or defensive caution.
    """
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return f"LLM Analysis unavailable. Error: {e}"

# ==========================================
# 3. DATA LOADING & CLEANING
# ==========================================
def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
    except:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')

    df.columns = df.columns.str.strip().str.replace('^[^a-zA-Z0-9]+', '', regex=True)
    print(f"✅ Successfully loaded: {filepath}")
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
    df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()
    
    numeric_cols = ['FTHG', 'FTAG', 'HST', 'AST', 'HY', 'AY', 'AvgH', 'AvgD', 'AvgA']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    df = df.sort_values('Date')
    return df

# ==========================================
# 4. STATS HELPERS
# ==========================================
def calculate_metrics(df):
    avg_h_goals = df['FTHG'].mean()
    avg_a_goals = df['FTAG'].mean()
    h_att = df.groupby('HomeTeam')['FTHG'].mean() / avg_h_goals
    h_def = df.groupby('HomeTeam')['FTAG'].mean() / avg_a_goals
    a_att = df.groupby('AwayTeam')['FTAG'].mean() / avg_a_goals
    a_def = df.groupby('AwayTeam')['FTHG'].mean() / avg_h_goals
    return h_att, h_def, a_att, a_def, avg_h_goals, avg_a_goals

def get_recent_stats(df, team, n=5):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(n)
    goals, sots, cards = [], [], []
    for _, row in recent.iterrows():
        if row['HomeTeam'] == team:
            goals.append(row['FTHG']); sots.append(row['HST']); cards.append(row['HY'])
        else:
            goals.append(row['FTAG']); sots.append(row['AST']); cards.append(row['AY'])
    return {"avg_g": np.mean(goals) if goals else 0, "avg_sot": np.mean(sots) if sots else 0, "avg_c": np.mean(cards) if cards else 0}

# ==========================================
# 5. PREDICTION ENGINE
# ==========================================
def run_predictor(home_team, away_team, df):
    h_att, h_def, a_att, a_def, avg_h, avg_a = calculate_metrics(df)
    
    h_stats = get_recent_stats(df, home_team)
    a_stats = get_recent_stats(df, away_team)
    league_avg_sot = df[['HST', 'AST']].mean().mean()

    # 1. Base Poisson Expectation
    home_exp = h_att[home_team] * a_def[away_team] * avg_h
    away_exp = a_att[away_team] * h_def[home_team] * avg_a

    if h_stats['avg_sot'] > (league_avg_sot * 1.15):
        home_exp *= 1.12
    if a_stats['avg_sot'] > (league_avg_sot * 1.15):
        away_exp *= 1.12

    if home_exp > (away_exp * 2.5):
        home_exp *= 1.25  
    elif away_exp > (home_exp * 2.5):
        away_exp *= 1.25

    rho = 0.05

    def get_tau(h, a, h_exp, a_exp, rho_val):
        if h == 0 and a == 0: return 1 - (h_exp * a_exp * rho_val)
        if h == 0 and a == 1: return 1 + (h_exp * rho_val)
        if h == 1 and a == 0: return 1 + (a_exp * rho_val)
        if h == 1 and a == 1: return 1 - rho_val
        return 1.0

    max_p, best_s = 0, (0, 0)
    for h in range(10):
        for a in range(10):
            p = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
            p *= get_tau(h, a, home_exp, away_exp, rho)
            
            if p > max_p:
                max_p, best_s = p, (h, a)
    
    # 2. Market Odds
    m = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team))].tail(1)
    odds_text = f"H: {m['AvgH'].iloc[0]} | D: {m['AvgD'].iloc[0]} | A: {m['AvgA'].iloc[0]}" if not m.empty else "No Recent Odds"

    # 3. History & Stats 
    h2h = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | 
             ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))].tail(5)
    
    h2h_summary = ""
    for _, row in h2h.iterrows():
        h2h_summary += f"{row['HomeTeam']} {int(row['FTHG'])}-{int(row['FTAG'])} {row['AwayTeam']}\n"

    # 4. PRINT DASHBOARD
    print(f"\n{'='*50}\nMATCHUP: {home_team} vs {away_team}\n{'='*50}")
    
    print(f"\n[HEAD-TO-HEAD HISTORY]")
    if h2h.empty:
        print("No past encounters found.")
    else:
        for _, row in h2h.sort_values('Date', ascending=False).iterrows():
            print(f"{row['Date'].strftime('%d/%m/%Y')}: {row['HomeTeam']} {int(row['FTHG'])}-{int(row['FTAG'])} {row['AwayTeam']}")

    print(f"\n[PROJECTION]")
    print(f"PROBABLE SCORE: {home_team} {best_s[0]} - {best_s[1]} {away_team}")
    print(f"EXPECTED GOALS: {home_team} ({home_exp:.2f}) - {away_team} ({away_exp:.2f})")
    print(f"MARKET CONTEXT: {odds_text}")
    
    print(f"\n[STRENGTH ANALYSIS]")
    print(f"{home_team}: {h_stats['avg_sot']:.1f} SOT | {h_stats['avg_c']:.1f} Avg Yellow Cards")
    print(f"{away_team}: {a_stats['avg_sot']:.1f} SOT | {a_stats['avg_c']:.1f} Avg Yellow Cards")
    
    # 5. CALL LLM 
    print("\n[THINKING]... Consulting AI Analyst...")
    insight = get_llm_explanation(home_team, away_team, best_s, h2h_summary, h_stats, a_stats, odds_text)
    print(f"\n[AI ANALYST INSIGHT]: {insight}")

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    dataset = load_and_clean_data("data-set.csv")
    all_teams = sorted(dataset['HomeTeam'].unique())
    
    print("\n--- Available Teams ---")
    print(", ".join(all_teams))
    
    while True:
        h_input = input("\nEnter Home Team (or 'exit'): ")
        if h_input.lower() == 'exit': break
        home = standardize_team_name(h_input, all_teams)
        if not home: print(f"❌ '{h_input}' not found."); continue
            
        a_input = input("Enter Away Team: ")
        away = standardize_team_name(a_input, all_teams)
        if not away: print(f"❌ '{a_input}' not found."); continue
            
        run_predictor(home, away, dataset)
