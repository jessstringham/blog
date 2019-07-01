---
title: 'Climate classification with Keras'
tags: [jupyter, project, quick]
layout: post
mathjax: true
display_image: 2019-06-02-ex.png
ipynb: https://github.com/jessstringham/notebooks/tree/master/2019-06-02-climate-with-keras.ipynb
---




One of my favorite hack projects was trying to create a [climate classification]({% post_url 2018-06-11-climate-classification-with-neural-nets %}) by clustering learned embeddings of weather stations.

The original model was written in TensorFlow. Since then, I've started to experiment with [Keras](https://keras.io). Because the climate classifier is pretty simple neural network, I rewrote the model using Keras and saved many lines of code.

For the problem description and data preparation, see the [original post]({% post_url 2018-06-11-climate-classification-with-neural-nets %}).




## Load the data

I'm using the same files that I created for the [original post]({% post_url 2018-06-11-climate-classification-with-neural-nets %}).



{% highlight python %}
DATA_PATH = 'data/weather'

matrix_file = os.path.join(DATA_PATH, 'data/gsn-2000-TMAX-TMIN-PRCP.npz')

# column labels
STATION_ID_COL = 0
MONTH_COL = 1
DAY_COL = 2
VAL_COLS_START = 3
TMAX_COL = 3
TMIN_COL = 4
PRCP_COL = 5

with np.load(matrix_file) as npz_data:
    weather_data = npz_data['data'].astype(np.int32)

# I decided to switch over to using the day of the year instead of two 
# eh, this isn't perfect (it assumes all months have 31 days), but it helps differentiate 
# the first of the month vs the last. 
weather_data_day_of_year_data = 31 * (weather_data[:, MONTH_COL] - 1) + (weather_data[:, DAY_COL] - 1)
{% endhighlight %}






{% highlight python %}
station_id_data = weather_data[:, STATION_ID_COL].reshape(-1, 1)
weather_data_day_of_year_data = weather_data_day_of_year_data.reshape(-1, 1)
weather_prediction = weather_data[:, VAL_COLS_START:]
{% endhighlight %}




## Define the network

I'll use the same network as [before]({% post_url 2018-06-11-climate-classification-with-neural-nets %}).

![](/assets/2018-06-11-nn.png)



{% highlight python %}
# network parameters
BATCH_SIZE = 50
EMBEDDING_SIZE = 20
HIDDEN_UNITS = 40

# and classification parameters. How many climates I want.
CLUSTER_NUMBER = 6

# count how many stations there are
NUM_STATIONS = np.max(weather_data[:, STATION_ID_COL]) + 1
{% endhighlight %}






{% highlight python %}
station_id_input = Input(shape=(1,), name='station_id_input')
month_day_input = Input(shape=(1,), name='month_day_input')

# Feed the station id through the embedding. This embeddings variable
# is the whole point of this network!
embedded_stations = Embedding(
    output_dim=EMBEDDING_SIZE, 
    input_dim=NUM_STATIONS,
    name='embedded_stations'
)(station_id_input)

embedded_station_reshape = Reshape((EMBEDDING_SIZE,), name='embedded_station_reshape')(embedded_stations)

station_and_day = Concatenate(name='station_and_day')([embedded_station_reshape, month_day_input])

# Now build a little network that can learn to predict the weather
hidden = Dense(HIDDEN_UNITS, activation='relu', name='hidden')(station_and_day)

prediction = Dense(
    3,  # Output for each of the attributes of the weather prediction (max, min, precipitation)
    activation=None,  # don't use an activation on predictions
    name='prediction'
)(hidden)


model = Model(inputs=[station_id_input, month_day_input], outputs=prediction)

model.compile(optimizer='adam',
              loss='mean_squared_error')

model.summary()
{% endhighlight %}




    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    station_id_input (InputLayer)   (None, 1)            0                                            
    __________________________________________________________________________________________________
    embedded_stations (Embedding)   (None, 1, 20)        24360       station_id_input[0][0]           
    __________________________________________________________________________________________________
    embedded_station_reshape (Resha (None, 20)           0           embedded_stations[0][0]          
    __________________________________________________________________________________________________
    month_day_input (InputLayer)    (None, 1)            0                                            
    __________________________________________________________________________________________________
    station_and_day (Concatenate)   (None, 21)           0           embedded_station_reshape[0][0]   
                                                                     month_day_input[0][0]            
    __________________________________________________________________________________________________
    hidden (Dense)                  (None, 40)           880         station_and_day[0][0]            
    __________________________________________________________________________________________________
    prediction (Dense)              (None, 3)            123         hidden[0][0]                     
    ==================================================================================================
    Total params: 25,363
    Trainable params: 25,363
    Non-trainable params: 0
    __________________________________________________________________________________________________



{% highlight python %}
model.fit(
    [
        station_id_data,
        weather_data_day_of_year_data,
    ], 
    weather_prediction.reshape(-1, 3), 
    epochs=1,
)
{% endhighlight %}




    Epoch 1/1
    2417066/2417066 [==============================] - 178s 74us/step - loss: 4390.3217


There is less boilerplate in the Keras code compared to my TensorFlow implementation in the [original post]({% post_url 2018-06-11-climate-classification-with-neural-nets %}). I think it's cool that most of the code is doing work describing the network.

## Classify the embeddings

Finally, I run KMeans on the trained embeddings to assign a "climate".



{% highlight python %}
trained_embeddings = model.get_layer('embedded_stations').get_weights()[0]
{% endhighlight %}






{% highlight python %}
with open(os.path.join(DATA_PATH, 'data/stations') )as f:
    list_of_stations = [line.strip() for line in f.readlines()]

kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=0).fit(trained_embeddings)

# I can export the classification here:
for station, label in zip(list_of_stations, kmeans.labels_):
    #print('{}\t{}'.format(station, label))
    pass
{% endhighlight %}




![](/assets/2019-06-02-new-map.png)

I get similar climates as before. Since there's randomness in the neural network initializations and batches and in the KMeans, I wouldn't expect to get exactly the same. For example, the latitude boundaries on the East coast have shifted compared to before.

For this post, I also switched to [Cartopy](https://github.com/SciTools/cartopy) from [Basemap](https://matplotlib.org/basemap/users/intro.html#cartopy-new-management-and-eol-announcement).