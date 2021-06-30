# import flask
from flask import Flask, render_template, redirect, request, make_response, jsonify

# import stock_data
import pandas_datareader.data as web
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import datetime

from sklearn.decomposition import PCA 
# import Normalizer
from sklearn.preprocessing import Normalizer
# import machine learning libraries
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

import trading_calendars as tc
import pytz
import time
import plotly
import plotly.express as px
import json

# import random

# Create an instance of Flask
app=Flask(__name__)

sym_list=['AMZN', 'AAPL', 'WBA', 'NOC', 'BA', 'LMT', 'MCD', 'INTC', 'NEE', 'IBM', 
		  'TXN', 'MA', 'MSFT', 'GE', 'GOOGL', 'AXP', 'PEP', 'KO', 'JNJ', 'GM', 
		  'HCA', 'AMGN', 'JPM', 'NFLX', 'UNH', 'V', 'VMC', 'VZ', 'LNC', 'WM', 
		  'TGT', 'PLD', 'CVX', 'PXD', 'PGR', 'NUE', 'TJX', 'MMM', 'MDT', 'LLY', 
		  'MAS', 'KR', 'ABC', 'AMAT', 'DE', 'UPS', 'LEN', 'WHR', 'ADBE', 'CE']

@app.route("/")
def home():
	return render_template("index.html")

@app.route('/stocks', methods=['GET', 'POST'])
def stocks(): 
	start_date=request.form.get('startdate')
	end_date=request.form.get('enddate')
	start_time=time.time()
	movements_df=get_data(random.choice(sym_list, 40), start_date, end_date)
	print(f'Fetch data took {time.time()-start_time} seconds. ')
	start_time=time.time()
	result=cluster(movements_df)
	print(f'Fetch data took {time.time()-start_time} seconds. ')
	
	fig=px.scatter_3d(result, x=0, y=1, z=2, color='Cluster', text='Symbol', width=800, height=800, opacity=1)
	fig.update(layout_coloraxis_showscale=False)
	fig.update_layout(
	    title='Overview',
	    scene=dict(
	        xaxis=dict(
	            title_text='PCA 1', 
	            showticklabels=False),
	        yaxis=dict(
	            title_text='PCA 2', 
	            showticklabels=False),
	        zaxis=dict(
	            title_text='PCA 3', 
	            showticklabels=False),
	        )
	    )

	graphJS=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('index.html', output=result[['Symbol', 'Cluster']].sort_values('Cluster').to_html(index=False), graph=graphJS)

def get_data(sym_list, start_date, end_date):
	data_source='yahoo'
	start_time=time.time()
	panel_data=web.DataReader(sym_list, data_source, start_date, end_date)
	panel_data.to_csv('output/original.csv')
	stock_open=np.array(panel_data['Open']).T
	stock_close=np.array(panel_data['Close']).T
	row, col=stock_close.shape
	movements=np.zeros([row, col])
	for i in range(0, row):
		movements[i,:]=np.subtract(stock_close[i,:], stock_open[i,:])

	xnys=tc.get_calendar("XNYS")
	date=xnys.sessions_in_range(
		# pd.Timestamp("2011-05-20", tz=pytz.UTC),
		# pd.Timestamp("2021-05-20", tz=pytz.UTC)
		pd.Timestamp(start_date, tz=pytz.UTC), 
		pd.Timestamp(end_date, tz=pytz.UTC)
	)
	movements_df=pd.DataFrame(movements, index=sym_list, columns=date)
	columns=list(movements_df.columns)
	new_columns=[]
	for each_column in columns: 
	    new_columns.append(each_column.tz_convert(None))#.replace('00:00:00+00:00', ''))
	movements_df.columns=new_columns
	movements_df.to_csv('output/movements_df.csv')
	# return sym_list, movements
	return movements_df

def cluster(movement_df):
    normalizer=Normalizer()
    inertia=[]
    current_inertia=0
    movement=normalizer.fit_transform(movement_df)
    reduced_data=PCA(n_components=3).fit_transform(movement)
    for i in range(2, 15): 
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(reduced_data)
        inertia.append(kmeans.inertia_)
    # plt.plot(list(range(2, 15)), inertia)
    diff=[inertia[i]-inertia[i-1] for i in range(1, len(inertia))]
    min_diff=diff[0]
    best_k=0
    for i in range(len(diff)): 
        each_diff=diff[i]
        if each_diff<min_diff: 
            print(f'Found lowest diff :{each_diff}')
            print(f'Best K: {i+2}')
            break
        else: 
            min_diff=each_diff
    # plt.plot(range(3, 15), diff)
    kmeans=KMeans(n_clusters=i+2)
    kmeans.fit(reduced_data)
    labels=kmeans.predict(reduced_data)
    result=pd.DataFrame(zip(movement_df.index, labels), columns=['Symbol', 'Cluster'])
    result=pd.concat([result, pd.DataFrame(reduced_data)], axis=1)
    result.sort_values('Cluster')
    print(diff)
    return result

if __name__=='__main__':
	app.run(debug=True)