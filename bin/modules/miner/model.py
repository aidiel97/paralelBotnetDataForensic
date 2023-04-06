import uuid

labelModel = {
  'labelId': 'uuid',
  'labelName': 'string'
}

IpAddressModel = {
  'addressId': 'uuid',
  'IPaddress': 'string'
}

def main(model, listOfElement):
  listOfDict = []
  listOfKeys = list(model.keys())
  for element in listOfElement:
    record = {
      listOfKeys[0]:str(uuid.uuid4())
    }
    if type(element) == list:
      for index, comp in element:
        listOfKeys[index+1]:comp
    else:
      record[listOfKeys[1]] = element
    
    listOfDict.append(record)

  return listOfDict