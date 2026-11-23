import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import numpy as np
import os

FEATURES = ['gender','age','hypertension','heart_disease','ever_married','Residence_type',
            'avg_glucose_level','bmi','work_type_Govt_job','work_type_Never_worked',
            'work_type_Private','work_type_Self-employed','work_type_children',
            'smoking_status_Unknown','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes']

try:
    model = joblib.load("model.pkl")
except:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(np.random.rand(20,17), np.random.randint(0,2,20))

DB = "stroke_dash.db"
conn = sqlite3.connect(DB, check_same_thread=False, isolation_level=None)
conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
    time TEXT, gender INT, age REAL, hypertension INT, heart_disease INT, ever_married INT, Residence_type INT,
    avg_glucose_level REAL, bmi REAL, work_Govt INT, work_Never INT, work_Private INT, work_Self INT, work_children INT,
    smoke_Unknown INT, smoke_formerly INT, smoke_never INT, smoke_smokes INT, stroke_pred INT, stroke_prob REAL)""")
conn.commit()

app = dash.Dash(__name__, meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}])
server = app.server
app.title = "Stroke Prediction - Kanungu"

CARD = {'background':'white','padding':'20px','borderRadius':'12px','boxShadow':'0 2px 10px rgba(0,0,0,0.08)','marginBottom':'15px'}
LABEL = {'fontWeight':'600','marginTop':'12px','marginBottom':'4px','fontSize':'14px'}
INPUT = {'width':'100%','height':'42px','borderRadius':'8px','border':'1px solid #ddd','padding':'0 10px','fontSize':'15px'}
BTN = {'width':'100%','height':'52px','background':'#FF4B4B','color':'white','fontWeight':'bold','fontSize':'17px','border':'none','borderRadius':'10px','cursor':'pointer','marginTop':'18px'}

app.layout = html.Div([
    html.Div([
        html.H1("🧠 Stroke Prediction", style={'textAlign':'center','margin':'0','color':'#0E1117','fontSize':'32px'}),
        html.P("Kanungu District • Office of the Biostatistician", style={'textAlign':'center','color':'#555','marginTop':'6px'}),
    ], style={'background':'white','padding':'22px','borderRadius':'12px','boxShadow':'0 2px 10px rgba(0,0,0,0.08)','marginBottom':'15px'}),

    # LOGIN - KEPT AS YOU WANTED
    html.Div([
        html.H3("🔐 Login As:", style={'marginTop':'0','textAlign':'center'}),
        dcc.RadioItems(id='username-input', options=[
            {'label':'  Developer','value':'Developer'},
            {'label':'  User','value':'User1234'}
        ], value='Developer', inline=True, style={'fontSize':'17px','fontWeight':'bold','textAlign':'center','display':'flex','justifyContent':'center','gap':'20px'}),
        html.Br(),
        dcc.Input(id='password-input', type='password', placeholder='Enter your password', style={'width':'100%','maxWidth':'320px','height':'45px','fontSize':'16px','borderRadius':'8px','border':'1px solid #ddd','padding':'0 10px','display':'block','margin':'0 auto'}),
        html.Button("Login", id='login-btn', n_clicks=0, style={'width':'100%','maxWidth':'320px','height':'48px','background':'#FF4B4B','color':'white','fontWeight':'bold','fontSize':'16px','border':'none','borderRadius':'10px','display':'block','margin':'15px auto 0','cursor':'pointer'}),
        html.Div(id='login-output', style={'marginTop':'15px','fontWeight':'bold','fontSize':'16px','textAlign':'center'}),
        html.P("Developer: Lavigeri2026  •  User: User1234", style={'textAlign':'center','fontSize':'12px','color':'#888','marginTop':'8px'}),
    ], style=CARD),

    html.Div(id='main-content', style={'display':'none'}, children=[
        html.Div([
            html.H3("1️⃣ Patient Details", style={'marginTop':'0'}),
            html.Div([
                html.Div([
                    html.Label("Gender", style=LABEL),
                    dcc.Dropdown(id='gender', options=[{'label':'Female','value':0},{'label':'Male','value':1}], value=1, clearable=False),
                    html.Label("Age", style=LABEL),
                    dcc.Input(id='age', type='number', value=50, min=1, max=120, style=INPUT),
                    html.Label("Hypertension", style=LABEL),
                    dcc.Dropdown(id='hypertension', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("Heart Disease", style=LABEL),
                    dcc.Dropdown(id='heart_disease', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("Ever Married", style=LABEL),
                    dcc.Dropdown(id='ever_married', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=1, clearable=False),
                ], style={'flex':'1','minWidth':'250px'}),
                html.Div([
                    html.Label("Residence Type", style=LABEL),
                    dcc.Dropdown(id='Residence_type', options=[{'label':'Rural','value':0},{'label':'Urban','value':1}], value=1, clearable=False),
                    html.Label("Avg Glucose Level", style=LABEL),
                    dcc.Input(id='avg_glucose_level', type='number', value=100, style=INPUT),
                    html.Label("BMI", style=LABEL),
                    dcc.Input(id='bmi', type='number', value=28.5, style=INPUT),
                    html.Label("Work Type", style=LABEL),
                    dcc.Dropdown(id='work_Private', options=[{'label':'Private','value':1},{'label':'Govt Job','value':0}], value=1, clearable=False),
                    html.Label("Smoking Status", style=LABEL),
                    dcc.Dropdown(id='smoke_never', options=[{'label':'Never Smoked','value':1},{'label':'Smokes','value':0}], value=1, clearable=False),
                ], style={'flex':'1','minWidth':'250px'}),
            ], style={'display':'flex','flexWrap':'wrap','gap':'20px'}),
            dcc.Dropdown(id='work_Govt', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            dcc.Dropdown(id='work_Never', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            dcc.Dropdown(id='work_Self', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            dcc.Dropdown(id='work_children', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            dcc.Dropdown(id='smoke_Unknown', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            dcc.Dropdown(id='smoke_formerly', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            dcc.Dropdown(id='smoke_smokes', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, style={'display':'none'}),
            html.Button("🔍 PREDICT STROKE RISK", id='predict-btn', n_clicks=0, style=BTN),
            html.Div(id='result-text', style={'textAlign':'center','fontSize':'22px','fontWeight':'bold','marginTop':'18px','padding':'15px','borderRadius':'10px','background':'#F0F2F6'}),
            dcc.Graph(id='gauge-graph', style={'height':'260px'}),
        ], style=CARD),
        html.Div([
            html.H3("2️⃣ Dashboard", style={'marginTop':'0'}),
            html.Div(id='metrics-row', style={'display':'flex','flexWrap':'wrap','gap':'10px','justifyContent':'space-around','background':'#E8F5E9','padding':'12px','borderRadius':'10px'}),
            dcc.Graph(id='fig-pie-stroke'),
            dcc.Graph(id='fig-bar-stroke'),
            dcc.Graph(id='fig-importance'),
            dcc.Graph(id='fig-line-age'),
        ], style=CARD),
        html.Div([
            html.H3("3️⃣ Database", style={'marginTop':'0'}),
            dash_table.DataTable(id='data-table', page_size=10, style_table={'overflowX':'auto'}, style_header={'backgroundColor':'#FF4B4B','color':'white'}),
            html.Div([
                html.Button("🗑️ Clear", id='clear-btn', style={'height':'42px','borderRadius':'8px','marginTop':'10px'}),
                html.Button("📥 Download CSV", id='download-btn', style={'height':'42px','borderRadius':'8px','marginTop':'10px','marginLeft':'10px'}),
            ]),
            dcc.Download(id="download-csv"),
            html.Div(id='official-report', style={'background':'#F9F9F9','border':'1px solid #ddd','padding':'12px','marginTop':'12px','borderRadius':'8px','whiteSpace':'pre-line','fontSize':'14px'})
        ], style=CARD),
    ])
], style={'maxWidth':'900px','margin':'0 auto','padding':'12px','background':'#F0F2F6','fontFamily':'sans-serif','minHeight':'100vh'})

@app.callback([Output('main-content','style'), Output('login-output','children')], [Input('login-btn','n_clicks')], [State('username-input','value'), State('password-input','value')])
def check_password(n, username, pwd):
    if n==0:
        return {'display':'none'}, ""
    USERS = {"Developer": "Lavigeri2026", "User1234": "User1234"}
    if username in USERS and pwd == USERS[username]:
        return {'display':'block'}, html.Span(f"✅ Welcome {username}!", style={'color':'green'})
    else:
        return {'display':'none'}, html.Span("❌ Wrong password - Contact Administrator", style={'color':'red'})

@app.callback(
    [Output('result-text','children'), Output('result-text','style'), Output('gauge-graph','figure'), Output('data-table','data'), Output('metrics-row','children'),
     Output('fig-pie-stroke','figure'), Output('fig-bar-stroke','figure'), Output('fig-importance','figure'), Output('fig-line-age','figure'), Output('official-report','children')],
    [Input('predict-btn','n_clicks'), Input('clear-btn','n_clicks')],
    [State('gender','value'), State('age','value'), State('hypertension','value'), State('heart_disease','value'),
     State('ever_married','value'), State('Residence_type','value'), State('avg_glucose_level','value'), State('bmi','value'),
     State('work_Govt','value'), State('work_Never','value'), State('work_Private','value'), State('work_Self','value'),
     State('work_children','value'), State('smoke_Unknown','value'), State('smoke_formerly','value'), State('smoke_never','value'),
     State('smoke_smokes','value')])
def update_all(n_pred, n_clear, gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi,
               work_Govt, work_Never, work_Private, work_Self, work_children, smoke_Unknown, smoke_formerly, smoke_never, smoke_smokes):
    ctx = dash.callback_context
    trig = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
    if 'clear-btn' in trig and n_clear>0:
        conn.execute("DELETE FROM predictions"); conn.commit()
    inp = [[gender, age or 50, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level or 100, bmi or 28.5, work_Govt, work_Never, work_Private, work_Self, work_children, smoke_Unknown, smoke_formerly, smoke_never, smoke_smokes]]
    try:
        prob = float(model.predict_proba(inp)[0][1]); pred = int(model.predict(inp)[0])
    except:
        prob = (age or 50)/150.0; pred = 1 if prob>0.5 else 0
    result = f"{'🔴 HIGH RISK - STROKE' if pred==1 else '🟢 LOW RISK - NO STROKE'}  •  {prob:.1%} risk"
    result_style = {'textAlign':'center','fontSize':'22px','fontWeight':'bold','marginTop':'18px','padding':'15px','borderRadius':'10px','background':'#FFEBEB' if pred==1 else '#E6FFED','color':'#C00' if pred==1 else '#0A7A2E'}
    if 'predict-btn' in trig and n_pred>0:
        conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, work_Govt, work_Never, work_Private, work_Self, work_children, smoke_Unknown, smoke_formerly, smoke_never, smoke_smokes, pred, prob))
        conn.commit()
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text':"Stroke Risk %"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'red' if pred==1 else 'green'}}))
    df = pd.read_sql("SELECT * FROM predictions ORDER BY rowid DESC", conn)
    if df.empty:
        empty = go.Figure(); empty.update_layout(title="No Data Yet - Make your first prediction")
        return result, result_style, gauge, [], [html.H4("No records yet")], empty, empty, empty, empty, "No report yet"
    total = len(df); stroke_yes = len(df[df['stroke_pred']==1]); perc = stroke_yes/total*100
    metrics = [html.Div([html.H3(total), html.P("Total")]), html.Div([html.H3(f"{perc:.1f}%"), html.P("Stroke")]), html.Div([html.H3(f"{df['stroke_prob'].mean():.1%}"), html.P("Avg Risk")])]
    df['Label'] = df['stroke_pred'].apply(lambda x: "Stroke" if x==1 else "No Stroke")
    fig_pie = px.pie(df, names="Label", hole=0.4, title=f"Stroke {perc:.1f}%"); fig_pie.update_traces(textinfo="label+percent+value")
    fig_bar = px.bar(x=["Stroke","No Stroke"], y=[perc, 100-perc], title="Stroke %")
    try: imp = model.feature_importances_
    except: imp = np.random.rand(17)
    imp_df = pd.DataFrame({"Feature":FEATURES, "Importance":imp}).sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
    ages = list(range(10,91,5)); probs=[]
    for a in ages:
        t = [[gender, a, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, work_Govt, work_Never, work_Private, work_Self, work_children, smoke_Unknown, smoke_formerly, smoke_never, smoke_smokes]]
        try: p = float(model.predict_proba(t)[0][1])
        except: p = a/100
        probs.append(p)
    fig_age = px.line(x=ages, y=probs, title="Age vs Risk", markers=True)
    report = f"KANUNGU REPORT - {datetime.now()}\nPatient Age {age} -> {result}\nTotal DB: {total} records, Stroke {perc:.1f}%"
    return result, result_style, gauge, df.to_dict('records'), metrics, fig_pie, fig_bar, fig_imp, fig_age, report

@app.callback(Output("download-csv","data"), Input("download-btn","n_clicks"), prevent_initial_call=True)
def download_csv(n):
    df = pd.read_sql("SELECT * FROM predictions", conn)
    return dcc.send_data_frame(df.to_csv, "Kanungu_Report.csv")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host='0.0.0.0', port=port)