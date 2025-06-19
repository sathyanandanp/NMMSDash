# pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns

# %%
import pandas as pd
import os

file_path = 'nmms.csv'

# Check if file exists in current folder
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("CSV loaded successfully!")
    print(df.head())
else:
    print(f"File not found: {file_path}")


# %%
df.columns = df.columns.str.replace(' ', '_')

# %%
summary = df.groupby(['SCHOOL_NAME', 'COMMUNITY']).size().reset_index(name='total_students')


# %%
pivot = df.pivot_table(index='SCHOOL_NAME', columns='COMMUNITY', values='SLNo', aggfunc='count', fill_value=0)


# %% [markdown]
# 

# %%
# Add a total column
pivot['Total'] = pivot.sum(axis=1)

# Sort by total in descending order
pivot_sorted = pivot.sort_values('Total', ascending=False)


# %%
# Calculate values per school
student_count = df.pivot_table(index='SCHOOL_NAME', values='SLNo', aggfunc='count', fill_value=0)
avg_mat = df.pivot_table(index='SCHOOL_NAME', values='MAT', aggfunc='mean', fill_value=0)
avg_sat = df.pivot_table(index='SCHOOL_NAME', values='SAT', aggfunc='mean', fill_value=0)
avg_total = df.pivot_table(index='SCHOOL_NAME', values='TOTAL', aggfunc='mean', fill_value=0)

# Rename columns for clarity
student_count = student_count.rename(columns={'SLNo': 'Student_Count'})
avg_mat = avg_mat.rename(columns={'MAT': 'Avg_MAT'})
avg_sat = avg_sat.rename(columns={'SAT': 'Avg_SAT'})
avg_total = avg_total.rename(columns={'TOTAL': 'Avg_TOTAL'})

# Concatenate all results into a single DataFrame
combined = pd.concat([student_count, avg_mat, avg_sat, avg_total], axis=1)

# Sort by student count descending
combined = combined.sort_values('Student_Count', ascending=False)


# %%
# Calculate per-school metrics
student_count = df.groupby('SCHOOL_NAME')['SLNo'].count().rename('Student_Count')
max_total = df.groupby('SCHOOL_NAME')['TOTAL'].max().rename('Max_TOTAL')

# Only calculate Avg_TOTAL for schools with at least 2 students
avg_total = df.groupby('SCHOOL_NAME')['TOTAL'].agg(
    lambda x: x.mean() if len(x) >= 2 else None
)
avg_total.name = 'Avg_TOTAL'

# Combine into a single DataFrame
combined = pd.concat([student_count, max_total, avg_total], axis=1)

# Remove schools where Avg_TOTAL is None (less than 2 students)
combined = combined[combined['Avg_TOTAL'].notnull()]

# Normalize each metric (Min-Max scaling)
combined['Student_Count_Norm'] = (combined['Student_Count'] - combined['Student_Count'].min()) / (combined['Student_Count'].max() - combined['Student_Count'].min())
combined['Max_TOTAL_Norm'] = (combined['Max_TOTAL'] - combined['Max_TOTAL'].min()) / (combined['Max_TOTAL'].max() - combined['Max_TOTAL'].min())
combined['Avg_TOTAL_Norm'] = (combined['Avg_TOTAL'] - combined['Avg_TOTAL'].min()) / (combined['Avg_TOTAL'].max() - combined['Avg_TOTAL'].min())

# Calculate ranking score
combined['Ranking_Score'] = (
    0.35 * combined['Student_Count_Norm'] +
    0.35 * combined['Max_TOTAL_Norm'] +
    0.30 * combined['Avg_TOTAL_Norm']
)

# Sort by Ranking_Score in descending order
combined = combined.sort_values('Ranking_Score', ascending=False)


# %%
# Remove DISTRICT from combined if it exists
if 'DISTRICT' in combined.columns:
    combined = combined.drop(columns=['DISTRICT'])

# Merge DISTRICT info into combined DataFrame (do this only once)
school_district = df[['SCHOOL_NAME', 'DISTRICT']].drop_duplicates().set_index('SCHOOL_NAME')
combined = combined.merge(school_district, left_index=True, right_index=True)


# Get top 10 schools per district based on Ranking_Score
top10_per_district = (
    combined
    .sort_values(['DISTRICT', 'Ranking_Score'], ascending=[True, False])
    .groupby('DISTRICT')
    .head(10)
)


# %%
combined = combined.reset_index()


# %%
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import textwrap

# Assume 'combined' and 'df' are already prepared as in your code above

# Add 'All Districts' option
districts = sorted(combined['DISTRICT'].unique())
district_options = [{'label': d, 'value': d} for d in districts]
district_options.insert(0, {'label': 'All Districts', 'value': 'ALL'})

metrics = ['Ranking_Score', 'Student_Count', 'Max_TOTAL', 'Avg_TOTAL']
bar_colors = ['#4B8BBE', '#E06C75', '#98C379', '#FFD43B']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Top 10 Schools Dashboard"),
    dcc.Dropdown(
        id='district-dropdown',
        options=district_options,
        value='ALL',
        clearable=False
    ),
    html.Div([
        dcc.Graph(id='bar-ranking'),
        dcc.Graph(id='bar-student'),
        dcc.Graph(id='bar-max'),
        dcc.Graph(id='bar-avg'),
        dcc.Graph(id='community-pie')
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '40px'})
])

@app.callback(
    [Output('bar-ranking', 'figure'),
     Output('bar-student', 'figure'),
     Output('bar-max', 'figure'),
     Output('bar-avg', 'figure'),
     Output('community-pie', 'figure')],
    [Input('district-dropdown', 'value')]
)
def update_charts(selected_district):
    if selected_district == 'ALL':
        data = combined.copy()
        filtered = df.copy()
    else:
        data = combined[combined['DISTRICT'] == selected_district]
        filtered = df[df['DISTRICT'] == selected_district]

    # Top 10 schools
    top10 = data.sort_values('Ranking_Score', ascending=False).head(10)
    top10_avg = top10[top10['Student_Count'] >= 2]

    # Bar charts
    bar_figs = []
    for i, metric in enumerate(metrics):
        if metric == 'Avg_TOTAL':
            plot_data = top10_avg
        else:
            plot_data = top10
        labels = [textwrap.fill(str(name), 15) for name in plot_data['SCHOOL_NAME']]
        fig = go.Figure(go.Bar(
            x=plot_data[metric],
            y=labels,
            orientation='h',  # Horizontal bar
            marker_color=bar_colors[i],
            text=[f'{v:.2f}' for v in plot_data[metric]],
            textposition='outside'
        ))
        fig.update_layout(
            title=metric.replace('_', ' '),
            yaxis_title='School Name',
            xaxis_title=metric.replace('_', ' '),
            yaxis_tickfont=dict(size=10),
            margin=dict(l=200),
            width=1200,
            height=400
        )
        fig.update_yaxes(autorange="reversed")
        if not plot_data.empty:
            if metric == 'Ranking_Score':
                fig.update_xaxes(range=[0.2, plot_data[metric].max() * 1.1])
            elif metric in ['Max_TOTAL', 'Avg_TOTAL']:
                fig.update_xaxes(range=[60, plot_data[metric].max() * 1.1])
        bar_figs.append(fig)

    # Community-wise pie chart
    community_counts = filtered['COMMUNITY'].value_counts()
    pie_fig = go.Figure(go.Pie(
        labels=community_counts.index,
        values=community_counts.values,
        textinfo='label+percent+value',
        hole=0.3
    ))
    pie_fig.update_layout(title="Community-wise Total Students", height=400)

    return bar_figs[0], bar_figs[1], bar_figs[2], bar_figs[3], pie_fig

# %%
# For Render deployment
server = app.server

if __name__ == '__main__':
    app.run(debug=True)




