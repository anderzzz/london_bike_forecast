import requests
import json
from requests.auth import HTTPBasicAuth

import pandas as pd
from datetime import datetime
from datetime import timedelta

api_url_base = 'https://api.tfl.gov.uk/bikepoint'
id = '???'
passwd = '???'

def request_london():
    xx = requests.get(api_url_base, auth=HTTPBasicAuth(id, passwd))
    json.dump(open('test.json', 'w'), xx.json[0])

def make_time_interval(start_date, end_date, interval_size):

    x0 = datetime.strptime(start_date, '%Y/%m/%d')
    x1 = datetime.strptime(end_date, '%Y/%m/%d')
    time_jump = timedelta(minutes=interval_size)

    t_interval_id = 0
    t_outer_bound = x0
    vals = []
    while t_outer_bound < x1:
        t_inner_bound = t_outer_bound
        t_outer_bound += time_jump
        t_interval = [t_interval_id, t_inner_bound.strftime('%d/%m/%Y %H:%M'),
                                     t_outer_bound.strftime('%d/%m/%Y %H:%M')]
        vals.append(pd.Series(t_interval))
        t_interval_id += 1

    df_out = pd.DataFrame(vals)
    df_out = df_out.rename(columns={0:'t_interval_id', 1:'t_lower', 2:'t_upper'})
    df_out['t_lower'] = pd.to_datetime(df_out['t_lower'], dayfirst=True)
    df_out['t_upper'] = pd.to_datetime(df_out['t_upper'], dayfirst=True)

    return df_out

def massage_data_file(file_path, tinterval, interval_size):

    def _match_interval(s, ti):
        xx = ti.loc[ti['t_lower'] <= s]
        yy = xx.loc[xx['t_upper'] > s]
        return yy['t_interval_id'].tolist()[0]

    df_raw = pd.read_csv(file_path, index_col=0)
    df_raw['End Date'] = pd.to_datetime(df_raw['End Date'], dayfirst=True)
    df_raw['Start Date'] = pd.to_datetime(df_raw['Start Date'], dayfirst=True)

    date0 = df_raw['End Date'].tolist()[0]
    date0_id = _match_interval(date0, tinterval)
    date_low = tinterval.loc[tinterval['t_interval_id'] == date0_id]['t_lower'].tolist()[0]

    id_val = date0_id
    date_val = date_low
    for k, row in df_raw['End Date'].iteritems():
        td = row - date_val
        if td > timedelta(minutes=interval_size):
            id_val += 1
            date_val += timedelta(minutes=interval_size)
        print (id_val, td, row)
    raise RuntimeError

    df_raw['End_Date_TID'] = df_raw['End Date'].apply(_match_interval, **{'ti': tinterval})
    df_raw['Start_Date_TID'] = df_raw['Start Date'].apply(_match_interval, **{'ti': tinterval})
    print (df_raw)
    print (df_raw.columns)
    print (df_raw.to_csv('ppp.csv'))

def main():
    tinterval = make_time_interval('2015/01/01', '2016/01/01', 10)
    tinterval.to_csv('tinterval.csv')
    tinterval_tmp = tinterval.loc[tinterval['t_lower'] > datetime(2015,9,19)]
    tinterval_tmp = tinterval_tmp.loc[tinterval_tmp['t_lower'] < datetime(2015,10,5)]
    massage_data_file('10a Journey Data Extract 20Sep15-03Oct15.csv', tinterval_tmp, 10)

if __name__ == '__main__':
    main()
