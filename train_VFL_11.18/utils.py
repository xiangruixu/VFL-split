
import csv
# split data
def bm_split_data(data):
    data_len = data['y'].count()
    split1 = int(data_len * 0.8)
    train_data = data[:split1]
    test_data = data[split1:]
    return train_data,  test_data

def credit_split_data(data):
    data_len = data['Y'].count()
    split1 = int(data_len * 0.8)
    train_data = data[:split1]
    test_data = data[split1:]

    return train_data,  test_data
    
def vehicle_split_data(data):
    data_len =len(data)
    split1 = int(data_len * 0.8)
    train_data = data[:split1]
    test_data = data[split1:]

    return train_data,  test_data



def records_path(acti, dataset, timing_t0):
    csvFile_1 = open(f'Results/VFL_client1_{acti}_{dataset}_{timing_t0}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results/VFL_client2_{acti}_{dataset}_{timing_t0}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    # if dataset == 'bank_marketing':
    #     writer_1.writerow(
    #         ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
    #          'att14', 'att15', 'att16', 'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24', 'att25',
    #          'att26', 'att27', 'att28', 'att29', 'att30', 'att31', 'att32',
    #          'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40', 'att41', 'att42', 'att43', 'att44',
    #          'att45', 'att46', 'att47', 'att48',
    #          'att49', 'att50', 'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
    #          'att61', 'att62', 'att63', 'att64',
    #          'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7'])
    #     writer_2.writerow(
    #         ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
    #          'att14', 'att15', 'att16',
    #          'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24', 'att25', 'att26', 'att27', 'att28',
    #          'att29', 'att30', 'att31', 'att32',
    #          'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40', 'att41', 'att42', 'att43', 'att44',
    #          'att45', 'att46', 'att47', 'att48',
    #          'att49', 'att50', 'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
    #          'att61', 'att62', 'att63', 'att64',
    #          'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7'])

    # if dataset == 'credit':
    #     writer_1.writerow(
    #         ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
    #          'att14', 'att15', 'att16', 'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24', 'att25',
    #          'att26', 'att27', 'att28', 'att29', 'att30', 'att31', 'att32', 'att33', 'att34', 'att35', 'att36', 'att37',
    #          'att38', 'att39', 'att40', 'att41', 'att42', 'att43', 'att44', 'att45', 'att46', 'att47', 'att48', 'att49',
    #          'att50', 'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
    #          'att61', 'att62', 'att63', 'att64',
    #          'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12'])
    #     writer_2.writerow(
    #         ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
    #          'att14', 'att15', 'att16', 'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24', 'att25',
    #          'att26', 'att27', 'att28', 'att29', 'att30', 'att31', 'att32',
    #          'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40', 'att41', 'att42', 'att43', 'att44',
    #          'att45', 'att46', 'att47', 'att48',
    #          'att49', 'att50', 'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
    #          'att61', 'att62', 'att63', 'att64',
    #          'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11'])
    # if dataset == 'census':
    #     writer_1.writerow(
    #         ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
    #          'att14', 'att15', 'att16', 'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24', 'att25',
    #          'att26', 'att27', 'att28', 'att29', 'att30', 'att31', 'att32',
    #          'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40', 'att41', 'att42', 'att43', 'att44',
    #          'att45', 'att46', 'att47', 'att48',
    #          'att49', 'att50', 'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
    #          'att61', 'att62', 'att63', 'att64',
    #          'age','class of worker','industry code','occupation code','education','wage per hour','enrolled in edu inst last wk','marital status','major industry code','major occupation code',
    #          'race','hispanic origin','sex','member of a labor union','reason for unemployment','full or part time employment stat','capital gains','capital losses','divdends from stocks','tax filer status'])
    #     writer_2.writerow(
    #         ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
    #          'att14', 'att15', 'att16',
    #          'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24', 'att25', 'att26', 'att27', 'att28',
    #          'att29', 'att30', 'att31', 'att32',
    #          'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40', 'att41', 'att42', 'att43', 'att44',
    #          'att45', 'att46', 'att47', 'att48',
    #          'att49', 'att50', 'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
    #          'att61', 'att62', 'att63', 'att64',
    #          'region of previous residence','state of previous residence','detailed household and family stat','detailed household summary in household','migration code-change in msa','migration code-change in reg','migration code-move within reg','live in this house 1 year ago','migration prev res in sunbelt','num persons worked for employer',
    #          'family members under 18','country of birth father','country of birth mother','country of birth self','citizenship','own business or self employed','fill inc questionnaire for veterans admin','veterans benefits','weeks worked in year','year'])

    return writer_1, writer_2


def records_path_decision(acti, dataset, end_shadow):
    csvFile_1 = open(f'Results_decision/VFL_client1_{acti}_{dataset}_{end_shadow}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results_decision/VFL_client2_{acti}_{dataset}_{end_shadow}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    # if dataset == 'bank_marketing':
    #     writer_1.writerow(
    #         ['att1', 'att2', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7'])
    #     writer_2.writerow(
    #         ['att1', 'att2', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7'])

    # if dataset == 'credit':
    #     writer_1.writerow(
    #         ['att1', 'att2', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12'])
    #     writer_2.writerow(
    #         ['att1', 'att2', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11'])

    return writer_1, writer_2