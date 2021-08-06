"""
Reads in the results json files, print the model accuracy and save all results in dataframe csv. 
"""
import json
import glob
import os
import pandas as pd

ip_dir = 'results/'
files = glob.glob(ip_dir + '*.json')
files.sort(key=os.path.getmtime)

print('Files')
for ii, ff in enumerate(files):
    print(str(ii).ljust(5) + ff)

acc_k = ['test_acc_1', 'test_acc_10', 'test_acc_100']

print('\nOverall Accuracy') 
print('ID'.ljust(5) + 'Dataset'.ljust(10) + 'Model Name'.ljust(60) + '1%'.ljust(7) + '10%'.ljust(7) + '100%'.ljust(7))
results = []
for ii, ff in enumerate(files):
    data = json.load(open(ff))
    res = ''
    for kk in acc_k:
        res+= str(round(data['results'][kk], 2)).ljust(7)
        
    print(str(ii).ljust(5) + data['args']['dataset'].ljust(10) + data['args']['exp_name'].ljust(60) + res)
    results.append({'cur_time':data['args']['cur_time'], 'dataset':data['args']['dataset'],'exp_name':data['args']['exp_name'],'epochs':data['epoch'],
                    'backbone':data['args']['backbone'],'batch_size':data['args']['batch_size'],'im_res':data['args']['im_res'],
                    'test_acc_1':data['results']['test_acc_1'],'test_acc_10':data['results']['test_acc_10'],'test_acc_100':data['results']['test_acc_100']})

acc_bal_k = ['test_acc_bal_1', 'test_acc_bal_10', 'test_acc_bal_100']

print('\nBalanced Accuracy') 
print('ID'.ljust(5) + 'Dataset'.ljust(10) + 'Model Name'.ljust(60) + '1%'.ljust(7) + '10%'.ljust(7) + '100%'.ljust(7))
results_bal = []
for ii, ff in enumerate(files):
    data = json.load(open(ff))
    
    res = ''
    for kk in acc_bal_k:
        res+= str(round(data['results'][kk], 2)).ljust(7)
        
    print(str(ii).ljust(5) + data['args']['dataset'].ljust(10) + data['args']['exp_name'].ljust(60) + res)
    results_bal.append({'cur_time':data['args']['cur_time'],'dataset':data['args']['dataset'],'exp_name':data['args']['exp_name'],'epochs':data['epoch'],
                        'backbone':data['args']['backbone'],'batch_size':data['args']['batch_size'],'im_res':data['args']['im_res'],
                        'test_acc_bal_1':data['results']['test_acc_bal_1'],'test_acc_bal_10':data['results']['test_acc_bal_10'],'test_acc_bal_100':data['results']['test_acc_bal_100']})
result_df = pd.DataFrame(pd.DataFrame(results)).merge(pd.DataFrame(results_bal),
left_on=['cur_time','dataset','exp_name'],right_on=['cur_time','dataset','exp_name'])
result_df.to_csv(ip_dir + 'result_r1_ICCT.csv')
