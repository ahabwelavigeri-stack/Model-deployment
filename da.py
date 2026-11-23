import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import numpy as np

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

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.title = "Stroke Prediction - Kanungu"

app.layout = html.Div([
    html.Div([
        html.H1("🧠 STROKE PREDICTION MODEL", style={'textAlign':'center','color':'#1f77b4'}),
        html.P("Kanungu District - Office of the Biostatistician", style={'textAlign':'center'}),
    ], style={'background':'#f0f8ff','padding':'15px','borderRadius':'10px'}),

    # LOGIN AS DEVELOPER OR USER - YOU GIVE PASSWORD
    html.Div([
        html.H3("🔐 Login As:", style={'marginBottom':'10px'}),
        dcc.RadioItems(id='username-input', options=[
            {'label':'  Developer','value':'Developer'},
            {'label':'  User','value':'User1234'}
        ], value='Developer', inline=True, style={'fontSize':'18px','fontWeight':'bold'}),
        html.Br(),
        dcc.Input(id='password-input', type='password', placeholder='Enter your password', style={'width':'300px','height':'45px','fontSize':'16px'}),
        html.Button("Login", id='login-btn', n_clicks=0, style={'marginLeft':'10px','height':'45px','background':'#1f77b4','color':'white','fontWeight':'bold','width':'100px'}),
        html.Div(id='login-output', style={'marginTop':'15px','fontWeight':'bold','fontSize':'16px'}),
    ], style={'textAlign':'center','margin':'30px auto','background':'white','padding':'25px','borderRadius':'12px','border':'2px solid #1f77b4','maxWidth':'500px'}),

    html.Div(id='main-content', style={'display':'none'}, children=[
        html.Div([
            html.H2("1️⃣ PREDICTING PART"),
            html.Div([
                html.Div([
                    html.Label("gender"), dcc.Dropdown(id='gender', options=[{'label':'Female (0)','value':0},{'label':'Male (1)','value':1}], value=1, clearable=False),
                    html.Label("age"), dcc.Slider(id='age', min=1, max=100, value=50, marks={0:'0',50:'50',100:'100'}, tooltip={"placement":"bottom"}),
                    html.Div(id='age-output'),
                    html.Label("hypertension"), dcc.Dropdown(id='hypertension', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("heart_disease"), dcc.Dropdown(id='heart_disease', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                ], className="three columns", style={'padding':'10px','border':'1px solid #ddd','borderRadius':'8px'}),
                html.Div([
                    html.Label("ever_married"), dcc.Dropdown(id='ever_married', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=1, clearable=False),
                    html.Label("Residence_type"), dcc.Dropdown(id='Residence_type', options=[{'label':'Rural','value':0},{'label':'Urban','value':1}], value=1, clearable=False),
                    html.Label("avg_glucose_level"), dcc.Input(id='avg_glucose_level', type='number', value=100, style={'width':'100%'}),
                    html.Label("bmi"), dcc.Input(id='bmi', type='number', value=28.5, style={'width':'100%'}),
                    html.Label("work_type_Govt_job"), dcc.Dropdown(id='work_Govt', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                ], className="three columns", style={'padding':'10px','border':'1px solid #ddd','borderRadius':'8px'}),
                html.Div([
                    html.Label("work_type_Private"), dcc.Dropdown(id='work_Private', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=1, clearable=False),
                    html.Label("smoking_status_never smoked"), dcc.Dropdown(id='smoke_never', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=1, clearable=False),
                    html.Label("smoking_status_smokes"), dcc.Dropdown(id='smoke_smokes', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Button("💾 PREDICT", id='predict-btn', n_clicks=0, style={'width':'100%','height':'50px','background':'#ff4d4d','color':'white','fontWeight':'bold'}),
                    html.H3(id='result-text', style={'textAlign':'center'}),
                    dcc.Graph(id='gauge-graph', style={'height':'250px'})
                ], className="three columns", style={'padding':'10px','border':'1px solid #ddd','borderRadius':'8px'}),
                html.Div([
                    html.Label("work_Never"), dcc.Dropdown(id='work_Never', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("work_Self"), dcc.Dropdown(id='work_Self', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("work_children"), dcc.Dropdown(id='work_children', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("smoke_Unknown"), dcc.Dropdown(id='smoke_Unknown', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                    html.Label("smoke_formerly"), dcc.Dropdown(id='smoke_formerly', options=[{'label':'No','value':0},{'label':'Yes','value':1}], value=0, clearable=False),
                ], className="three columns", style={'padding':'10px','border':'1px solid #ddd','borderRadius':'8px'}),
            ], className="row"),
        ]),
        html.Hr(),
        html.Div([
            html.H2("2️⃣ DASHBOARD"), 
            html.Div([html.Div([dcc.Graph(id='fig-pie-stroke')], className="six columns"), html.Div([dcc.Graph(id='fig-bar-stroke')], className="six columns")], className="row"),
            html.Div([html.Div([dcc.Graph(id='fig-importance')], className="six columns"), html.Div([dcc.Graph(id='fig-line-age')], className="six columns")], className="row"),
        ]),
        html.Hr(),
        html.Div([
            html.H2("3️⃣ DATABASE"),
            html.Div(id='metrics-row', style={'display':'flex','justifyContent':'space-around','background':'#e8f5e9','padding':'10px'}),
            dash_table.DataTable(id='data-table', page_size=10, style_table={'overflowX':'auto'}, style_header={'backgroundColor':'#1f77b4','color':'white'}),
            html.Button("🗑️ Clear", id='clear-btn'), html.Button("📥 Download CSV", id='download-btn', style={'marginLeft':'10px'}),
            dcc.Download(id="download-csv"),
            html.Div(id='official-report', style={'background':'white','border':'1px solid black','padding':'10px','marginTop':'10px','whiteSpace':'pre-line'})
        ])
    ])
])

@app.callback([Output('main-content','style'), Output('login-output','children')], [Input('login-btn','n_clicks')], [State('username-input','value'), State('password-input','value')])
def check_password(n, username, pwd):
    if n==0:
        return {'display':'none'}, ""
    USERS = {"Developer": "Lavigeri2026", "User1234": "User1234"}
    if username in USERS and pwd == USERS[username]:
        return {'display':'block'}, html.Span(f"✅ Welcome {username}!", style={'color':'green'})
    else:
        return {'display':'none'}, html.Span("❌ Wrong password - Contact Administrator", style={'color':'red'})

@app.callback(Output('age-output','children'), Input('age','value'))
def show_age(a): return f"Age: {a}"

@app.callback(
    [Output('result-text','children'), Output('gauge-graph','figure'), Output('data-table','data'), Output('metrics-row','children'),
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
    inp = [[gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, work_Govt, work_Never, work_Private, work_Self, work_children, smoke_Unknown, smoke_formerly, smoke_never, smoke_smokes]]
    try:
        prob = float(model.predict_proba(inp)[0][1]); pred = int(model.predict(inp)[0])
    except:
        prob = age/150.0; pred = 1 if prob>0.5 else 0
    result = f"{'🔴 STROKE' if pred==1 else '🟢 NO STROKE'} {prob:.1%}"
    if 'predict-btn' in trig and n_pred>0:
        conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, work_Govt, work_Never, work_Private, work_Self, work_children, smoke_Unknown, smoke_formerly, smoke_never, smoke_smokes, pred, prob))
        conn.commit()
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text':"Risk %"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'red' if pred==1 else 'green'}}))
    df = pd.read_sql("SELECT * FROM predictions ORDER BY rowid DESC", conn)
    if df.empty:
        empty = go.Figure(); empty.update_layout(title="No Data Yet")
        return result, gauge, [], [html.H4("No records")], empty, empty, empty, empty, "No report yet"
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
    return result, gauge, df.to_dict('records'), metrics, fig_pie, fig_bar, fig_imp, fig_age, report

@app.callback(Output("download-csv","data"), Input("download-btn","n_clicks"), prevent_initial_call=True)
def download_csv(n):
    df = pd.read_sql("SELECT * FROM predictions", conn)
    return dcc.send_data_frame(df.to_csv, "Kanungu_Report.csv")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)