#!/users/animalecs/anaconda3/envs/py39/bin/python

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import dash
import plotly.graph_objects as go
from dash import dcc, html, Input, Output

# Load data
file_path = "demand.parquet"  # Adjust as needed
df = pd.read_parquet(file_path)

# Convert date from milliseconds to datetime
df['date'] = pd.to_datetime(df['date'], unit='ms')

# Get unique product codes for dropdown selection
unique_products = df['prod_code'].dropna().unique().tolist()

# Parameters
initial_stock = 2000
lead_time = 4  # Giorni
order_day = "Thursday"
lookback_period = 30  # Giorni usati per il calcolo del riordino
Z_score = 1.64  # 95% service level

def calculate_safety_stock(demand_series, lookback_period=30, lead_time=4, Z=1.64):
    """ Calcola lo stock di sicurezza basato sulla variabilità della domanda. """
    std_demand = demand_series['demand'].rolling(lookback_period, min_periods=1).std()
    return Z * std_demand.iloc[-1] * np.sqrt(lead_time)  # Safety Stock Formula


def predict_demand(demand_series, lookback_period=30, lead_time=4):
    """ Calculate reorder quantity based on moving average of past demand. """
    avg_demand = demand_series['demand'].rolling(lookback_period, min_periods=1).mean()
    return avg_demand.iloc[-1] * lead_time  # Forecast demand for lead time

def simulate_replenishment(demand_series, strategy):
    stock = initial_stock
    orders = []
    pending_orders = []
    results = []
    
    for i, row in demand_series.iterrows():
        date, demand = row['date'], row['demand']
        stock -= demand  # Deduct demand
        if stock < 0:
            stock = 0
        
        # Process incoming orders FIRST
        for arrival, qty in pending_orders:
            if arrival <= date:
                stock += qty
        
        # Remove only the orders that have been fulfilled
        pending_orders = [(arrival, qty) for arrival, qty in pending_orders if arrival > date]

        
        # Calculate reorder quantity dynamically
        reorder_quantity = predict_demand(demand_series[demand_series['date'] <= date], lookback_period, lead_time)
        safety_stock = calculate_safety_stock(demand_series[demand_series['date'] <= date], lookback_period, lead_time, Z_score)
        
        # Evita nuovi ordini se ne è già stato inviato uno in attesa di arrivo
        if len(pending_orders) > 0:
            results.append({'date': date, 
                            'stock': stock, 
                            'demand': demand, 
                            'orders_placed': len(orders), 
                            'reorder_point': reorder_quantity, 
                            'safety_stock': safety_stock})
            continue

        # Strategy-specific order logic
        if strategy == 1:
            # Order every Thursday if below ROP
            if stock < reorder_quantity and date.strftime('%A') == order_day:
                orders.append((date, reorder_quantity))
                pending_orders.append((date + timedelta(days=lead_time), reorder_quantity))
        elif strategy == 2:
            # Order immediately if below ROP
            if stock < reorder_quantity:
                orders.append((date, reorder_quantity))
                pending_orders.append((date + timedelta(days=lead_time), reorder_quantity))
        elif strategy == 3:
            if stock < safety_stock or (stock < reorder_quantity and date.strftime('%A') == order_day):
                orders.append((date, reorder_quantity))
                pending_orders.append((date + timedelta(days=lead_time), reorder_quantity))
        
        
        results.append({'date': date, 
                        'stock': stock, 
                        'demand': demand, 
                        'orders_placed': len(orders), 
                        'reorder_point': reorder_quantity, 
                        'safety_stock': safety_stock})
    
    return pd.DataFrame(results)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simulazione di Riordino Magazzino"),
    html.P("Questa dashboard mostra l'andamento delle scorte in base a diverse strategie di riordino."),
    html.P("I giovedì sono segnati con una X sul grafico."),
    html.P("In tutte le strategie non mandiamo un altro ordine se il precedente non è arrivato."),
    html.P("Le formule utilizzate sono:"),
    html.Ul([
        html.Li("Punto di Riordino: media della domanda negli ultimi 30 giorni moltiplicata per il lead time."),
        html.Li("Stock di Sicurezza: Z-score * deviazione standard della domanda * radice quadrata del lead time."),
    ]),
    html.P("Le tre strategie simulate sono:", style={'font-weight': 'bold'}),
    html.Ul([
        html.Li("Riordino ogni giovedì se le scorte sono sotto la soglia di riordino."),
        html.Li("Riordino immediato se le scorte scendono sotto il livello di Safety Stock calcolato."),
    ]),
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': str(p), 'value': str(p)} for p in unique_products],
        value=unique_products[0] if unique_products else None,
        clearable=False
    ),
    
    # Add Loading Component
    dcc.Loading(
        id="loading-1",
        type="default",  # Options: "default", "circle", "dot"
        children=[
            dcc.Graph(id='simulation-graph-1'),
            # dcc.Graph(id='simulation-graph-2'),
            dcc.Graph(id='simulation-graph-3'),
        ]
    ),
    
    html.Hr(),
    html.P("Codice sviluppato da Alex Mina - [GitHub](https://github.com/animalecs)", style={'text-align': 'center'})
])

server = app.server  # Necessario per Railway

@app.callback(
    [Output('simulation-graph-1', 'figure'),
     # Output('simulation-graph-2', 'figure'),
     Output('simulation-graph-3', 'figure')],
    Input('product-dropdown', 'value')
)
def update_graphs(selected_product):
    product_data = df[df['prod_code'] == selected_product].sort_values(by='date')
    demand_series = product_data[['date', 'demand']].copy()
    demand_series['demand'] = demand_series['demand'].fillna(0)
    start_date = demand_series['date'].min() + timedelta(days=lookback_period)
    demand_series = demand_series[demand_series['date'] >= start_date]
    
    sim1 = simulate_replenishment(demand_series, strategy=1)
    # sim2 = simulate_replenishment(demand_series, strategy=2)
    sim3 = simulate_replenishment(demand_series, strategy=3)
    
    # Filter last two years for charts
    end_date = demand_series['date'].max()
    start_date = end_date - timedelta(days= 365)
    
    sim1_filtered = sim1[sim1['date'] >= start_date]
    # sim2_filtered = sim2[sim2['date'] >= start_date]
    sim3_filtered = sim3[sim3['date'] >= start_date]
    
    # Create separate figures for each strategy
    def create_figure(sim, title, include_safety_stock=True):
        y_columns = ['stock', 'demand', 'reorder_point']
        
        # Remove safety_stock from Chart 1
        if include_safety_stock:
            y_columns.append('safety_stock')
        
        fig = px.line(sim, x='date', y=y_columns, title=f"{title}")

        # Customize line styles
        for trace in fig.data:
            if trace.name == 'reorder_point':
                trace.line.dash = 'dash'
                trace.line.width = 1
            elif trace.name == 'safety_stock' and include_safety_stock:
                trace.line.dash = 'dash'
                trace.line.width = 1
            else:
                trace.line.width = 2  # Keep stock and demand more visible

        # Generate ALL Thursdays within the dataset's time range
        start_date = sim['date'].min()
        end_date = sim['date'].max()
        
        # Create a list of all Thursdays between start_date and end_date
        all_thursdays = pd.date_range(start=start_date, end=end_date, freq='W-THU')

        # 🔹 Add "X" markers on all Thursdays
        fig.add_trace(go.Scatter(
            x=all_thursdays,
            y=[sim['stock'].min()] * len(all_thursdays),  # Place markers at the bottom
            mode='text',
            text="X",
            textposition="bottom center",
            marker=dict(color="red", size=12),
            showlegend=False
        ))

        # Stockout days annotation
        # Stockout days annotation (bold & highlighted)
        stockout_days = (sim['stock'] == 0).sum()
        fig.add_annotation(
            x=sim['date'].iloc[-1], y=sim['stock'].iloc[-1],
            text=f"<b>Stockout Days: {stockout_days}</b>",
            showarrow=False,
            font=dict(color='white', size=16, family="Arial Black"),
            align="center",
            bgcolor="red",
            bordercolor="black",
            borderwidth=2
        )

        return fig
    
    return (
        create_figure(sim1_filtered, "Riordino calendarizzato giovedì (no safety stock)", False),
       #create_figure(sim2_filtered, "Immediate Reorders"),
        create_figure(sim3_filtered, "Riordino calendarizzato giovedì o sotto scorta sicurezza")
    )

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=5001)
