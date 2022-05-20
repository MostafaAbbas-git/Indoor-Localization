from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import logging
import pymongo
from datetime import datetime
import time


logging.basicConfig(filename='record.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

logging.getLogger('flask_cors').level = logging.DEBUG
DATABASE_URL = "mongodb+srv://iotuser:i6BR3FhLxiagRhxO@iot-task0.slr9i.mongodb.net/iot-task0?retryWrites=true&w=majority"

# establish connection with database
client = pymongo.MongoClient(DATABASE_URL)

mongo_db = client.task1  # assign database to mongo_db

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "<h1>Indoor Localization task (#3)</h1>"


@app.route('/model', methods=['GET', 'POST'])
def model():

    model = pickle.load(open('./app/classifier.sav', 'rb'))

    if request.method == 'GET':
        """Return the latest location from the database with previousLocations list"""

        results_data = []
        reversed_results_data = []

        results_timestamp = []
        reversed_results_timestamp = []

        results = mongo_db.location.find(
            sort=[('_id', pymongo.DESCENDING)]).limit(50)

        for item in (results):
            results_data.append(item['data'])

            # Export the time and send it
            timestamp = item.get('_id').generation_time
            local_time = utc2local(timestamp)
            time = local_time.strftime("%H:%M:%S")

            results_timestamp.append(time)

        for reversed_item in reversed(results_data):
            reversed_results_data.append(reversed_item)

        for reversed_timestamp in reversed(results_timestamp):
            reversed_results_timestamp.append(reversed_timestamp)

        latest_location = reversed_results_data[-1]

        reversed_results_data.pop(-1)
        reversed_results_timestamp.pop(-1)

        response = jsonify(
            {"data": latest_location,
             "prevLocation": reversed_results_data,
             "timestamp": reversed_results_timestamp}
        )

        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    if request.method == 'POST':
        """Update the location"""

        # Expect the data to be in json format
        content = request.get_json(force=True)

        signalValues = np.array([content['values']])
        # Send the values to the model
        model_results = model.predict(signalValues)
        new_location = int(model_results[0])

        # Fetch the latest added location and compare it to the newly added one

        fetched_results = mongo_db.location.find().sort([('_id', -1)]).limit(1)
        latest_location = fetched_results[0]['data']

        if new_location != latest_location:
            mongo_db.location.insert_one({'data': new_location})
            response = jsonify(
                {"message": "Location updated successfully.", "location": new_location})

            response.headers.add('Access-Control-Allow-Origin', '*')
            app.logger.info(f'New Location updated with: {new_location}')
            return response

        else:
            response = jsonify(
                {"message": "No change found in the location."})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


@app.route('/droplocation', methods=['DELETE'])
def drop_location():
    collist = mongo_db.list_collection_names()

    if "location" in collist:
        # drop the collection
        mongo_db.location.drop()
    # create new collection
    mydb = client["task1"]
    mycol = mydb["location"]
    # insert 0 into the new collection
    mydict = {'data': 0}
    mycol.insert_one(mydict)
    return "Database cleared successfully"


@app.route('/hassan', methods=['POST', 'GET'])
def hassan():
    # Expect the data to be in json format
    content = request.get_json(force=True)

    input = content['input']

    response = jsonify({"response": int(input)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def utc2local(utc):
    epoch = time.mktime(utc.timetuple())
    offset = datetime.fromtimestamp(epoch) - datetime.utcfromtimestamp(epoch)
    return utc + offset
