
import json 
__author__ = 'yscoder@foxmail.com'

train_data_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/news150w/news2016zh_train.json"
train_data_target_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/news150w/news2016zh_train.text"
fin = open(train_data_target_path, 'w')
train_samples = 10000 
train_iter = 0 
fin.write('fuck')
# fin.close() 
lines = []
with open(train_data_path, 'r') as f:

    for json_line in f:
        json_data = json.loads(json_line)
        # print(json_data['content'])
        print(type(json_data['content']))
        lines.append(json_data['content'])
        train_iter += 1 
        if train_iter > train_samples:
            break
fin.writelines(lines)
fin.close()