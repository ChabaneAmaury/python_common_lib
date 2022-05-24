import gc
import tensorflow as tf
import numpy as np
import mongodbAPI


def load_mean_var(mongoClient, x, y):
    def field(v):
        return f"${v}"

    mean_var_query = [
        {
            "$unwind": {
                "path": field(x),
                "includeArrayIndex": "i"
            }
        },
        {
            "$group": {
                "_id": "$i",
                "totalDocuments": {
                    "$sum": 1
                },
                "sum": {
                    "$sum": field(x)
                },
                "std": {
                    "$stdDevPop": field(x)
                }
            }
        },
        {
            "$project": {
                "totalDocuments": "$totalDocuments",
                "mean": {
                    "$divide": [
                        "$sum",
                        "$totalDocuments"
                    ]
                },
                "var": {
                    "$pow": [
                        "$std",
                        2
                    ]
                }
            }
        },
        {
            "$sort": {
                "_id": 1
            }
        },
        {
            "$group": {
                "_id": None,
                "avg": {
                    "$push": "$mean"
                },
                "var": {
                    "$push": "$var"
                }
            }
        },
        {
            "$addFields": {
                "total_x": {
                    "$size": "$avg"
                }
            }
        }
    ]

    total_y_query = [
        {
            "$project": {
                "total": {
                    "$size": field(y)
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "total_y": {
                    "$avg": "$total"
                }
            }
        }
    ]

    cursor = list(mongoClient.collection.aggregate(mean_var_query))[0]
    mean = cursor["avg"]
    var = cursor["var"]
    total_x = cursor["total_x"]
    cursor = list(mongoClient.collection.aggregate(total_y_query))[0]
    total_y = cursor["total_y"]

    return mean, var, total_x, total_y


class MongoDBGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, db=None, collection=None, batch_size=16, shuffle=True):
        self.__IDs = None

        if db is not None and collection is not None:
            self.db = db
            self.collection = collection
        else:
            raise IOError('A MongoDB database is required')

        self.batch_size = batch_size
        self.x = x
        self.y = y

        mongoClient = mongodbAPI.MongodbAPI(db=self.db, collection=self.collection)
        self.n = mongoClient.collection.count_documents({})
        mongoClient.client.close()

        self.shuffle = shuffle
        self.indexes = np.arange(self.n)

    def getIds(self):
        if self.__IDs is None:
            mongoClient = mongodbAPI.MongodbAPI(db=self.db, collection=self.collection)
            ids = mongoClient.collection.aggregate([
                {
                    "$group": {
                        "_id": None,
                        "ids": {
                            "$push": "$_id"
                        }
                    }
                }
            ])
            self.__IDs = ids.next()["ids"]
            mongoClient.client.close()
        return self.__IDs

    def __transform_x(self, v):
        return np.array(v)

    def __transform_y(self, v):
        return np.array(v, dtype=np.uint8) / 255.0

    def on_epoch_end(self):
        self.indexes = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getdata(self, list_IDs):
        batch_x = []
        batch_y = []
        mongoClient = mongodbAPI.MongodbAPI(db=self.db, collection=self.collection)
        q = mongoClient.collection.find({
            "_id": {"$in": list_IDs},
        }, {
            self.x: 1,
            self.y: 1
        })
        for v in q:
            batch_x.append(self.__transform_x(v[self.x]))
            batch_y.append(self.__transform_y(v[self.y]))
        mongoClient.client.close()
        return batch_x, batch_y

    def __getitem__(self, index):
        assert self.db is not None and self.collection is not None
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        list_IDs = [self.getIds()[v] for v in indexes]
        batch_x, batch_y = self.__getdata(list_IDs)
        gc.collect()
        return np.array(batch_x), np.array(batch_y)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))
