import pandas as pd







def process_device_info(json_data,headers):


    json_trim = json_data[1:-1]
    json_keys_values = json_trim.replace('"','').split(',')

    dict = {}

    for val in json_keys_values:
        dict[val.split(':')[0].strip()] = str(val.split(':')[1].strip())

    line_out = ''
    for header in headers:
        if header.strip() in dict:
            line_out = line_out + dict[header.strip()] + ','
        else:
            line_out = line_out + '' + ','

    return line_out[:-1]


def main():

    device_headers = ["browser",
                   "browserVersion",
                   "browserSize",
                   "operatingSystem",
                   "operatingSystemVersion",
                    "isMobile",
                   "mobileDeviceBranding",
                   "mobileDeviceModel",
                   "mobileInputSelector",
                    "mobileDeviceInfo",
                   "mobileDeviceMarketingName",
                    "flashVersion",
                   "language",
                   "screenColors",
                   "screenResolution",
                   "deviceCategory"]

    geonetwork_headers = ["continent",
                          "subContinent",
                          "country",
                          "region",
                          "metro",
                          "city",
                          "cityId",
                          "networkDomain",
                          "latitude",
                          "longitude",
                          "networkLocation"]

    totals_headers = ["visits",
                      "hits",
                      "pageviews",
                      "bounces",
                      "newVisits",
                      "transactionRevenue"]

    traffic_source_headers = ["campaign",
                              "source",
                              "medium",
                              "keyword",
                              "adwordsClickInfo.criteriaParameters",
                              "isTrueDirect",
                              "referralPath",
                              "adwordsClickInfo.page",
                              "adwordsClickInfo.slot",
                              "adwordsClickInfo.gclId",
                              "adwordsClickInfo.adNetworkType",
                              "adwordsClickInfo.isVideoAd",
                              "adContent",
                              "campaignCode"]


    #train_file = '/home/nitin/Desktop/google_analytics/all/train.csv'
    test_file = '/home/nitin/Desktop/google_analytics/all/test.csv'

    data = pd.read_csv(test_file,low_memory=False)

    file_headers = []
    file_headers.append('channelGrouping')
    file_headers.append('date')

    for val in device_headers:
        file_headers.append('device_' + val)
    file_headers.append('fullVisitorId')

    for val in geonetwork_headers:
        file_headers.append('geoNetwork_' + val)
    file_headers.append('sessionId')
    file_headers.append('socialEngagementType')

    for val in totals_headers:
        file_headers.append('totals_' + val)

    for val in traffic_source_headers:
        file_headers.append('trafficSource_' + val)

    file_headers.append('visitId')
    file_headers.append('visitNumber')
    file_headers.append('visitStartTime')

    print (file_headers)
    index = None
    row = None



    with open('/home/nitin/Desktop/google_analytics/all/_flattest.csv','w') as flatf:
        flatf.writelines(",".join(file_headers))
        flatf.write('\n')
        for index, row in data.iterrows():

            try:

                line_main = row[0] + ',' + str(row[1])
                line_main = line_main + ',' + process_device_info(row[2],device_headers)
                line_main = line_main + ',' + str(row[3])
                line_main = line_main + ',' + process_device_info(row[4], geonetwork_headers)
                line_main = line_main + ',' + str(row[5])
                line_main = line_main + ',' + str(row[6])
                line_main = line_main + ',' + process_device_info(row[7], totals_headers)
                line_main = line_main + ',' + process_device_info(row[8], traffic_source_headers)
                line_main = line_main + ',' + str(row[9])
                line_main = line_main + ',' + str(row[10])
                line_main = line_main + ',' + str(row[11])

                #print (line_main)
                flatf.writelines(line_main)
                flatf.write('\n')

            except Exception as e:
                print ('The offensive index is:' + str(index))
                print ('The offensive row:' + str(row))

    #process_device_info(None,device_headers)


main()