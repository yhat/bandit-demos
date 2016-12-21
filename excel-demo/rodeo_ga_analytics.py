from googleapiclient.errors import HttpError
from googleapiclient import sample_tools
from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from apiclient.discovery import build
import pandas as pd
from datetime import date
import os

# Authenticate and create the service for the Core Reporting API
credentials = ServiceAccountCredentials.from_json_keyfile_name(
  '/Users/coristig/Downloads/My Project-e038d19ed699.json', ['https://www.googleapis.com/auth/analytics.readonly'])
http_auth = credentials.authorize(Http())
service = build('analytics', 'v3', http=http_auth)

# Yhat domain 67218338
# https://analytics.google.com/analytics/web/?hl=en#report/content-event-events/a37140626w65425131p67218338/%3F_r.drilldown%3Danalytics.eventAction%3Adownload%2Canalytics.eventCategory%3ARodeo-backend%26explorer-graphOptions.selected%3Danalytics.nthDay%26explorer-graphOptions.primaryConcept%3Danalytics.uniqueEventsTrue%26explorer-table.plotKeys%3D%5B%5B%22Linux%22%5D%2C%5B%22Macintosh%22%5D%2C%5B%22Windows%22%5D%5D/
def get_downloads(service):
  return service.data().ga().get(
    ids='ga:67218338',
    start_date='90daysAgo',
    end_date='yesterday',
    metrics='ga:uniqueEvents',
    dimensions='ga:date,ga:eventLabel',
    sort='ga:date',
    filters='ga:eventAction==download;ga:eventCategory==Rodeo'
    )

def get_downloads_with_os_version(service):
  return service.data().ga().get(
    ids='ga:67218338',
    start_date='90daysAgo',
    end_date='yesterday',
    metrics='ga:uniqueEvents',
    dimensions='ga:date,ga:operatingSystem,ga:operatingSystemVersion',
    sort='ga:date',
    filters='ga:eventAction==download;ga:eventCategory==Rodeo',
    start_index='1'
    )

# Rodeo Domain: 125637421
def get_total_and_new_users(service):
    return service.data().ga().get(
    ids='ga:125637421',
    start_date='90daysAgo',
    end_date='yesterday',
    metrics='ga:newusers,ga:users',
    dimensions='ga:date',
    sort='ga:date',
    start_index='1'
    )

def get_new_users(service):
  return service.data().ga().get(
    ids='ga:125637421',
    start_date='90daysAgo',
    end_date='yesterday',
    metrics='ga:newusers',
    dimensions='ga:date,ga:operatingSystem',
    sort='ga:date',
    start_index='1'
    )

def get_new_users_with_version(service):
  return service.data().ga().get(
    ids='ga:125637421',
    start_date='90daysAgo',
    end_date='yesterday',
    metrics='ga:newusers',
    dimensions='ga:date,ga:operatingSystem,ga:operatingSystemVersion',
    sort='ga:date',
    start_index='1'
    )

def new_users_command_result(service):
  return service.data().ga().get(
    ids='ga:125637421',
    start_date='90daysAgo',
    end_date='yesterday',
    metrics='ga:sessions',
    dimensions='ga:date,ga:operatingSystem',
    sort='ga:date',
    filters='ga:eventAction==execute_result',
    segment='gaid::-2',
    start_index='1'
    )

downloads = get_downloads(service).execute()
downloads_ver = get_downloads_with_os_version(service).execute()
users = get_total_and_new_users(service).execute()

new_users = get_new_users(service).execute()
new_users_ver = get_new_users_with_version(service).execute()
new_users_cmd = new_users_command_result(service).execute()

dl = pd.DataFrame(downloads['rows'], columns=['date','OS','downloads'])
dl.OS = dl.OS.map(lambda x: 'Windows' if(x.find('Windows')==0) else x)
dl.OS = dl.OS.map(lambda x: 'Macintosh' if(x.find('OS X')==0) else x)
dl.downloads = dl.downloads.astype(int)
dl = dl.groupby(['date','OS'], sort=False, as_index=False).sum()

dlv = pd.DataFrame(downloads_ver['rows'], columns=['date','OS','version','downloads'])

nu = pd.DataFrame(new_users['rows'], columns=['date','OS','new_users'])
nuv = pd.DataFrame(new_users_ver['rows'], columns=['date','OS','version','new_users'])
nuc = pd.DataFrame(new_users_cmd['rows'], columns=['date','OS','new_user_cmds'])


loss = pd.merge(dl, nu, on=['date','OS'])
loss = pd.merge(loss, nuc, on=['date','OS'])

loss.downloads = loss.downloads.astype('int')
loss.new_users = loss.new_users.astype('int')
loss.new_user_cmds = loss.new_user_cmds.astype('int')

loss['download_loss'] = loss['downloads'] - loss['new_users']
loss['new_user_loss'] = loss['new_users']- loss['new_user_cmds']
loss['total_loss'] = loss['downloads'] - loss['new_user_cmds']

loss.date = pd.DatetimeIndex(loss['date'])

loss_g = loss
loss_g['week'] = pd.DatetimeIndex(loss_g['date']).weekofyear
loss_w = loss_g.groupby(['week','OS'], sort=False,as_index=False)['downloads','new_users','new_user_cmds','download_loss','new_user_loss','total_loss'].sum()


# Create a Pandas Excel writer using XlsxWriter as the engine.
xlname = date.today().strftime("%d-%m-%y") + '_rodeo.xlsx'
writer = pd.ExcelWriter(xlname, engine='xlsxwriter')
workbook = writer.book

for OS in ['Macintosh', 'Windows', 'Linux']: 
    print(OS)
    loss_w[loss_w.OS==OS].to_excel(writer, index=False, sheet_name=OS)
    loss_w[loss_w.OS==OS].to_excel(writer, index=False, sheet_name=OS)
    loss_w[loss_w.OS==OS].to_excel(writer, index=False, sheet_name=OS)

#     worksheet = writer.sheets[OS]

# Configure the series of the chart from the dataframe data.
os.chdir('/job/output-files/')
writer.save()

# https://analytics.google.com/analytics/web/?hl=en#report/visitors-cohort/a37140626w120093807p125637421/%3FcohortTab-cohortOption.hasLoaded%3Dtrue%26cohortTab-cohortOption.granularity%3DWEEKLY%26cohortTab-cohortOption.dateRange%3D6%26cohortTab-cohortOption.selectedMetric%3Danalytics.cohortRetentionRate%26cohortTab-cohortOption.selectedDimension%3Danalytics.firstVisitDate%26_.useg%3DusertysdwRIsQZ60HMps_Zaehg%2CuserrxDFsCL0S7yim8aYyygRDw/


