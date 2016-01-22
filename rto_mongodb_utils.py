''' This set of functions lets you manipulate the mongoDB database
'''

def add_lowercase_metadata_staintype(db):
    '''
    :return: returns nothing
    '''
    for d in db.find({"metadata.stain_type":{"$exists": True}}):
        db.update({"_id":d["_id"]},{"$set":{"metadata.stain_type_lower":d["metadata"]["stain_type"].lower()}})

def add_lowercase_group_name(db):
    '''
    :return: returns nothing
    '''
    for d in db.find({"group.name":{"$exists": True}}):
        db.update({"_id":d["_id"]},{"$set":{"group.name":d["group"]["name"].lower()}})
