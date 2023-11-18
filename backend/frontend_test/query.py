import requests,json
import pandas as pd


end_point = 'http://127.0.0.1:8000/backend/random/'

#token = 'eb35de7a6cea41824b251ab9b8e55dc0f36495bc'
#token = '094462db0a6846182b012813d57fd2aab76ccd56'
token = '12c97617f766829fb669acc663602c8442f9271e'
# data = json.dumps(data)
headers = {
    'Authorization': 'Token {}'.format(token),
     'Content-Type': 'application/json;charset=UTF-8',
     'DN':'100'
}
response = requests.get(url=end_point,headers=headers)
print(response)
data = json.loads(response.text)
status = response
df = pd.DataFrame(data)
print('data is {} and status is {}'.format(data,status))
print(df.head())