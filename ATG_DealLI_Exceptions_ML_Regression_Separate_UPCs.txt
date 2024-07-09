from pandas import set_option,concat,DataFrame,Series,to_numeric,options
from numpy import concatenate,round
from pyodbc import connect
from warnings import filterwarnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,max_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor#,plot_tree
from sklearn.model_selection import GridSearchCV,train_test_split#,cross_val_score
from sklearn.preprocessing import StandardScaler#,LabelBinarizer,label_binarize,MaxAbsScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import *
from re import search#,match
from random import randint
from time import time
from smtplib import SMTP_SSL
from ssl import create_default_context
from os import path,walk
from zipfile import ZipFile, ZIP_DEFLATED
from time import time,sleep

filterwarnings('ignore')
set_option('display.max_rows', None)
set_option('display.max_columns', None)

server = 'MOE'
database = 'prod_WeisMarkets_RecoverNow'
connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Integrated Security={True};Autocommit={True};Trusted_Connection=yes;'
conn = connect(connectionString)
cursor_1 = conn.cursor()

def export_to_sql(data: DataFrame):
    server = 'barney'
    database = 'sandbox_mp'
    connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Integrated Security={True};Autocommit={True};Trusted_Connection=yes;'
    conn = connect(connectionString)
    cursor = conn.cursor()
    for index in range(len(list(data.columns))):
        if(not(index in [202,201,200,199,198,197,196,195,181])):
            data[list(data.columns)[index]] = data[list(data.columns)[index]].astype(str)
        else:
            data[list(data.columns)[index]] = data[list(data.columns)[index]].astype(float)
            data[list(data.columns)[index]] = data[list(data.columns)[index]].round(4)
    insert_sql = """
        INSERT INTO Weis_Market_Claim_With_ML_Predictions_PdOI_ATG (
			ATG_RCPT_DTL_DS_Checked,
			ATG_RCPT_DTL_DE_Checked,
			ATG_RCPT_DTL_ClientFamilyID_ATG,
			ATG_RCPT_DTL_BusinessUnit,
			ATG_RCPT_DTL_ReceiptNbr,
			ATG_RCPT_DTL_ReceiptSfx,
			ATG_RCPT_DTL_PONbr,
			ATG_RCPT_DTL_PODate,
			ATG_RCPT_DTL_ReceiptDate,
			ATG_RCPT_DTL_POVendorNbr,
			ATG_RCPT_DTL_GrossWght,
			ATG_RCPT_DTL_UnitOfWeight,
			ATG_RCPT_DTL_POQty,
			ATG_RCPT_DTL_ReceiptQty,
			ATG_RCPT_DTL_ShortQty_ATG,
			ATG_RCPT_DTL_VarWghtInd,
			ATG_RCPT_DTL_ItemNbr,
			ATG_RCPT_DTL_ItemDescr,
			ATG_RCPT_DTL_Dept,
			ATG_RCPT_DTL_UPCNbr,
			ATG_RCPT_DTL_ItemSize,
			ATG_RCPT_DTL_CasePack,
			ATG_RCPT_DTL_TurnQty_ATG,
			ATG_RCPT_DTL_TurnUnitQty_ATG,
			ATG_RCPT_DTL_TurnRatio_ATG,
			ATG_RCPT_DTL_ListCostSB_ATG,
			ATG_RCPT_DTL_ChangeDate_ATG,
			ATG_RCPT_DTL_EndDate_ATG,
			ATG_RCPT_DTL_NextCost_ATG,
			ATG_RCPT_DTL_PrevCost_ATG,
			ATG_RCPT_DTL_CostDiff_ATG,
			ATG_RCPT_DTL_IncDecFlag_ATG,
			ATG_RCPT_DTL_CostType_ATG,
			ATG_RCPT_DTL_AddDate_ATG,
			ATG_RCPT_DTL_SellEffDate_ATG,
			ATG_RCPT_DTL_LastShipDate_ATG,
			ATG_RCPT_DTL_ATG_Cost_Ref,
			ATG_RCPT_DTL_PdGross_ATG,
			ATG_RCPT_DTL_PdNet_ATG,
			ATG_RCPT_DTL_PdQty_ATG,
			ATG_RCPT_DTL_PdUnitGross_ATG,
			ATG_RCPT_DTL_PdUnitNet_ATG,
			ATG_RCPT_DTL_PdUnitQty_ATG,
			ATG_RCPT_DTL_Facility,
			ATG_RCPT_DTL_ATG_Hdr_Ref,
			ATG_RCPT_DTL_SourceFile_ATG,
			ATG_RCPT_DTL_ATG_Ref,
			ATG_RCPT_DTL_ATG_ItemID,
			ATG_RCPT_DTL_RcptFacility,
			ATG_RCPT_DTL_APVendorNbr,
			ATG_RCPT_DTL_Merchndsr,
			ATG_RCPT_DTL_ShipWght,
			ATG_RCPT_DTL_ShipCube,
			ATG_RCPT_DTL_PalltQty,
			ATG_RCPT_DTL_QualifyAmt,
			ATG_RCPT_DTL_RepckRatio,
			ATG_RCPT_DTL_Status,
			ATG_RCPT_DTL_ExcptnDate,
			ATG_RCPT_DTL_OKDate,
			ATG_RCPT_DTL_AdjCde1,
			ATG_RCPT_DTL_AdjQty1,
			ATG_RCPT_DTL_AdjCde2,
			ATG_RCPT_DTL_AdjQty2,
			ATG_RCPT_DTL_ListCost,
			ATG_RCPT_DTL_OI,
			ATG_RCPT_DTL_FreeGds,
			ATG_RCPT_DTL_WhseDisc,
			ATG_RCPT_DTL_UpDnAmt,
			ATG_RCPT_DTL_UpDnInd,
			ATG_RCPT_DTL_BB,
			ATG_RCPT_DTL_LastCost,
			ATG_RCPT_DTL_LastCost_Orig,
			ATG_RCPT_DTL_CashDisc,
			ATG_RCPT_DTL_FrghtAllw,
			ATG_RCPT_DTL_FrghtAllwInd,
			ATG_RCPT_DTL_VndrUpDn,
			ATG_RCPT_DTL_VndrUpDnInd,
			ATG_RCPT_DTL_PPayAdd,
			ATG_RCPT_DTL_PrePayAddExInd,
			ATG_RCPT_DTL_FrghtBill,
			ATG_RCPT_DTL_Bckhl,
			ATG_RCPT_DTL_VarWght,
			ATG_RCPT_DTL_InvQty,
			ATG_RCPT_DTL_InvListCost,
			ATG_RCPT_DTL_InvOI,
			ATG_RCPT_DTL_InvFreeGds,
			ATG_RCPT_DTL_InvWhseDisc,
			ATG_RCPT_DTL_InvUpDnAmt,
			ATG_RCPT_DTL_InvUpDnInd,
			ATG_RCPT_DTL_InvBB,
			ATG_RCPT_DTL_InvLastCost,
			ATG_RCPT_DTL_InvLastCost_Orig,
			ATG_RCPT_DTL_InvCashDisc,
			ATG_RCPT_DTL_InvFrghtAllw,
			ATG_RCPT_DTL_InvFrghtAllwExInd,
			ATG_RCPT_DTL_InvVndrUpDn,
			ATG_RCPT_DTL_InvVndrUpDnInd,
			ATG_RCPT_DTL_InvPPayAdd,
			ATG_RCPT_DTL_InvPPayAddExInd,
			ATG_RCPT_DTL_InvFrghtBill,
			ATG_RCPT_DTL_InvBckhl,
			ATG_RCPT_DTL_InvWght,
			ATG_RCPT_DTL_InvVarWght,
			ATG_RCPT_DTL_AdjQty,
			ATG_RCPT_DTL_AdjListCost,
			ATG_RCPT_DTL_AdjOI,
			ATG_RCPT_DTL_AdjFreeGds,
			ATG_RCPT_DTL_AdjWhseDisc,
			ATG_RCPT_DTL_AdjUpDnAmt,
			ATG_RCPT_DTL_AdjBB,
			ATG_RCPT_DTL_AdjLastCost,
			ATG_RCPT_DTL_AdjLastCost_Orig,
			ATG_RCPT_DTL_AdjFrghtAllw,
			ATG_RCPT_DTL_AdjVndrUpDn,
			ATG_RCPT_DTL_AdjPPayAdd,
			ATG_RCPT_DTL_AdjFrghtBill,
			ATG_RCPT_DTL_AdjBckhl,
			ATG_RCPT_DTL_AdjWght,
			ATG_RCPT_DTL_AdjVarWght,
			ATG_RCPT_DTL_AdjUpDnInd,
			ATG_RCPT_DTL_AdjVndrUpDnInd,
			ATG_RCPT_DTL_Toggle,
			ATG_RCPT_DTL_LastRcvCorrDate,
			ATG_RCPT_DTL_Comment,
			ATG_RCPT_DTL_FreeCs,
			ATG_RCPT_DTL_Trans,
			ATG_RCPT_DTL_TransQty,
			ATG_RCPT_DTL_Trans2RcvNbr,
			ATG_RCPT_DTL_Trans2RcvSfx,
			ATG_RCPT_DTL_Trans2Qty,
			ATG_RCPT_DTL_TransFromRcvNbr,
			ATG_RCPT_DTL_TransFromRcvSfx,
			ATG_RCPT_DTL_TransFromQty,
			ATG_RCPT_DTL_ItmFrtAllwExInd,
			ATG_RCPT_DTL_AdjPPayAddExInd,
			ATG_RCPT_DTL_FrghtBillExInd,
			ATG_RCPT_DTL_InvFrghtBillExInd,
			ATG_RCPT_DTL_AdjFrghtBillExInd,
			ATG_RCPT_DTL_BckhlExInd,
			ATG_RCPT_DTL_InvBckhlExInd,
			ATG_RCPT_DTL_AdjBckhlExInd,
			ATG_RCPT_DTL_DealFlg,
			ATG_RCPT_DTL_DealStatus,
			ATG_RCPT_DTL_InvQualifyAmt,
			ATG_RCPT_DTL_InvFreeCs,
			ATG_RCPT_DTL_AdjQualifyAmt,
			ATG_RCPT_DTL_AdjFreeCs,
			ATG_RCPT_DTL_CsUnitFctr,
			ATG_RCPT_DTL_InvSurchgInd,
			ATG_RCPT_DTL_HiOldAvgCost,
			ATG_RCPT_DTL_ACChgTkn,
			ATG_RCPT_DTL_ExcptnUsrID,
			ATG_RCPT_DTL_OKUsrID,
			ATG_RCPT_DTL_WDCostFlg,
			ATG_RCPT_DTL_OICostFlg,
			ATG_RCPT_DTL_FGCostFlg,
			ATG_RCPT_DTL_FACostFlg,
			ATG_RCPT_DTL_PPAddCostFlg,
			ATG_RCPT_DTL_BBCostFlg,
			ATG_RCPT_DTL_UDCostFlg,
			ATG_RCPT_DTL_VndUDCostFlg,
			ATG_RCPT_DTL_FBCostFlg,
			ATG_RCPT_DTL_BHCostFlg,
			ATG_RCPT_DTL_FullCostFlg,
			ATG_RCPT_DTL_DSDItmDueVndr,
			ATG_RCPT_DTL_ATG_Ref_Orig,
			ATG_RCPT_DTL_ItemFacility,
			ATG_RCPT_DTL_ATG_PO_Ref,
			ATG_RCPT_DTL_IncvOI_ATG,
			ATG_RCPT_DTL_IncvBB_ATG,
			ATG_RCPT_DTL_FlatAmtOI_ATG,
			ATG_RCPT_DTL_FlatAmtBB_ATG,
			ATG_RCPT_DTL_Contact,
			ATG_RCPT_DTL_BestOI_ATG,
			ATG_RCPT_DTL_BestBB_ATG,
			ATG_RCPT_DTL_UPCUnit,
			ATG_RCPT_DTL_BalFlag_ATG,
			ATG_RCPT_DTL_OOB_ATG,
			ATG_RCPT_DTL_PdHdrFrtVar_ATG,
			ATG_RCPT_DTL_MiscAdj_ATG,
			ATG_RCPT_DTL_PdOI_ATG_BeforeAI,
			ATG_RCPT_DTL_PdBB_ATG,
			ATG_RCPT_DTL_PdUpDn_ATG,
			ATG_RCPT_DTL_PdIncvOI_ATG,
			ATG_RCPT_DTL_PdFlatOI_ATG,
			ATG_RCPT_DTL_PdFlatBB_ATG,
			ATG_RCPT_DTL_PdBckHl_ATG,
			ATG_RCPT_DTL_PdPPayAdd_ATG,
			ATG_RCPT_DTL_PdFrtBil_ATG,
			ATG_RCPT_DTL_PdFrtAlw_ATG,
			ATG_RCPT_DTL_PdShortOI_ATG,
			ATG_RCPT_DTL_PdShortBB_ATG,
			ATG_RCPT_DTL_PdShortBckHl_ATG,
			Linear_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			DecisionTree_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			RandomForest_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			XGBoost_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			GradientBoost_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			LGBM_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			CatBoost_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG,
			KNeighbors_Model_Predictions_ATG_RCPT_DTL_PdOI_ATG
        )
        VALUES 
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?) 
    """
    for row in data.index:
        if(int(row)%100==0):
            print(f"{row:,}")
        current_row = list(data.iloc[int(row)])
        # for item in current_row:
        #     print(f"{item}: {type(item)}")
        # del current_row[33]
        # for index in range(len(current_row)):
        #     if(index>=33):
        #         current_row[index] = round(current_row[index],4)
        #     if(index in [2,3,4]):
        #         current_row[index] = int(current_row[index])
        #     else:
        #         current_row[index] = str(current_row[index])
        # current_row.insert(33,current_row[23])
        # del current_row[23]
        try:
            cursor.execute(insert_sql, current_row)
            conn.commit()
        except:
            pass
    cursor.close()
    conn.close()

def scatter_plot_models(target: Series,predictions: Series,model_name: str, show: bool=False):
    plt.figure(figsize=(6,5))
    plt.scatter(target,predictions,color='blue',label='Test_Data',linewidths=.1)
    plt.plot([Series(target).min(),Series(target).max()],
            [Series(target).min(),Series(target).max()],'k--',lw=1)
    plt.xlabel('PdOI_ATG_Before_AI')
    plt.ylabel('PdOI_ATG_After_AI')
    plt.savefig(f"C:/Code/Python/Machine_Learning_AI/Model_Analysis/{model_name}_Scatter.png")
    if(show):
        plt.show()
    else:
        plt.close()

def get_scores(target, predictions):
    mse = mean_squared_error(target, predictions)
    rmse = mse**0.5
    median_ae = median_absolute_error(target,predictions)
    mean_ae = mean_absolute_error(target,predictions)
    max_errors = max_error(target,predictions)
    r2 = r2_score(target,predictions)
    return [mse,rmse,mean_ae,median_ae,max_errors,r2]

def run_and_analyze_model(model,features_train_subset: DataFrame,target_train_subset: Series,features_test: Series,target_test: Series,model_scores: DataFrame,model_name: str):
    model.fit(features_train_subset,target_train_subset)
    train_predictions = Series(model.predict(features_train_subset),name=model_name)
    train_predictions = train_predictions.round(4)
    predictions = Series(model.predict(features_test),name=model_name)
    predictions = predictions.round(4)
    model_scores.loc[model_name] = get_scores(target_test,predictions)
    try:
        model_importances = DataFrame([model.feature_importances_],columns=features_train_subset.columns,index=[model_name]).T
        return model,train_predictions,predictions,model_scores,model_importances
    except:
        model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=[model_name]).T
        return model,train_predictions,predictions,model_scores,model_importances

def empty_string_to_null(string: str):
    if(len(str(string))==0 or len(str(string).replace(' ',''))==0):
        return "None"
    return string

def object_to_int(original,lookup_table):
    for row in lookup_table.index:
        for col in lookup_table.columns:
            if(original==lookup_table[col][row]):
                return int(col)

def random_int(min_val, max_val):
    # Get the current time in microseconds
    current_time = int(time() * 1000000)
    
    # Use the current time as a seed and perform some operations to get more randomness
    seed = (current_time ^ (current_time >> randint(1,20))) & 0xFFFFFFFF
    seed = (seed ^ (seed << randint(1,20))) & 0xFFFFFFFF
    seed = (seed ^ (seed >> randint(1,20))) & 0xFFFFFFFF
    
    # Scale the seed to the desired range
    random_value = min_val + (seed % (max_val - min_val + 1))
    
    return random_value

def random_features(columns):
    subset = []
    columns = list(columns)
    while(len(subset)<50):
        random_index = random_int(0,len(list(columns))-1)
        while(columns[random_index] in subset):
            random_index = random_int(0,len(list(columns))-1)
        subset.append(columns[random_index])
    return subset

def send_email_to_self(subject: str,body: str):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "micpowers98@gmail.com"  # Enter your address
    receiver_email = "micpowers98@gmail.com"  # Enter receiver address
    password = 'efex cwhv gppq ueob'
    message = "Subject: "+subject+"\n"+body

    context = create_default_context()
    with SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def get_data(columns: list):
	cursor_1.execute("""
			select 
				case when (li.checked='true' and ds.checked='claim') then 'Y' else 'N' end as IsClaimFlg, li.checked,
					li.[ATG_Ref],
					li.[ATG_LI_Ref],
					li.[BatchNbr],
					li.[CATEGORY_ATG],
					li.[ClaimType],
					li.[AP_VndNbr],
					li.[APVndrName],
					li.[OOB_ATG],
					li.[ItemNbr],
					li.[UPCNbr],
					li.[UPCUnit],
					li.[ItemDescription],
					li.[ItemShipPack],
					li.[PoNbr],
					li.[PODate],
					li.[ReceivingDate],
					li.[TurnRatio_ATG],
					li.[TurnQty_ATG],
					li.[OrdQty],
					li.[PdQty_ATG],
					li.[PdGross_ATG],
					li.[PdOI_ATG],
					li.[PdBB_ATG],
					li.[PdNet_ATG],
					li.[DealNbr],
					li.[OrdStartDate_ATG],
					li.[OrdEndDate_ATG],
					li.[DateStartArrival_ATG],
					li.[DateEndArrival_ATG],
					li.[DLAmtOI],
					li.[DLAmtBB]
			from     
				[MOE].[prod_WeisMarkets_RecoverNow].[dbo].[ATG_DealLI_Exceptions] li
						join  [MOE].[prod_WeisMarkets_RecoverNow].[dbo].[ATG_Deal_Summary] ds
						on (li.BatchNbr = ds.BatchNbr)
						and (li.DLVendorNbr = ds.DLVendorNbr)
						and (li.DealNbr = ds.DealNbr)
						and (li.CATEGORY_ATG = ds.CATEGORY_ATG)
						and (li.ClaimType = ds.ClaimType_ATG)
			where li.ClaimType = 'IN DEAL' and ds.CATEGORY_ATG = 'SAME VENDOR - AMT DEALS'
			order by 8,12,17			
		""")
	train_subset_data = cursor_1.fetchall()
	train_subset_data_list = []
	for index in range(len(train_subset_data)):
		train_subset_data_list.append(list(train_subset_data[index]))
	del train_subset_data
	return DataFrame(data=train_subset_data_list,columns=columns)

def get_big_data(columns: list):
	cursor_1.execute("""
            SELECT DS.checked as DS_Checked, DE.checked as DE_Checked, RD.*
                FROM 
                    MOE.prod_WeisMarkets_RecoverNow.dbo.ATG_RCPT_DTL as RD LEFT OUTER JOIN
                        MOE.prod_WeisMarkets_RecoverNow.dbo.ATG_DealLI_Exceptions_TBL as DE 
                        ON RD.ATG_Ref = DE.ATG_LI_Ref LEFT OUTER JOIN
                        MOE.prod_WeisMarkets_RecoverNow.dbo.ATG_Deal_Summary_TBL as DS
                            ON DE.ClaimType = DS.ClaimType_ATG
                                AND DE.DealNbr = DS.DealNbr
                                AND DE.DLVendorNbr = DS.DLVendorNbr
                                AND DE.BatchNbr = DS.BatchNbr		
		""")
	train_subset_data = cursor_1.fetchall()
	train_subset_data_list = []
	for index in range(len(train_subset_data)):
		train_subset_data_list.append(list(train_subset_data[index]))
	del train_subset_data
	return DataFrame(data=train_subset_data_list,columns=columns)

def get_individual_upc(columns: list, upc: str):
    cursor_1.execute(f"""
            SELECT DS.checked as DS_Checked, DE.checked as DE_Checked, RD.*
                FROM 
                    MOE.prod_WeisMarkets_RecoverNow.dbo.ATG_RCPT_DTL as RD LEFT OUTER JOIN
                        MOE.prod_WeisMarkets_RecoverNow.dbo.ATG_DealLI_Exceptions_TBL as DE 
                        ON RD.ATG_Ref = DE.ATG_LI_Ref LEFT OUTER JOIN
                        MOE.prod_WeisMarkets_RecoverNow.dbo.ATG_Deal_Summary_TBL as DS
                            ON DE.ClaimType = DS.ClaimType_ATG
                                AND DE.DealNbr = DS.DealNbr
                                AND DE.DLVendorNbr = DS.DLVendorNbr
                                AND DE.BatchNbr = DS.BatchNbr
                WHERE
                    RD.UPCNbr={upc}
		""")
    train_subset_data = cursor_1.fetchall()
    train_subset_data_list = []
    for index in range(len(train_subset_data)):
        train_subset_data_list.append(list(train_subset_data[index]))
    del train_subset_data
    #print(DataFrame(data=train_subset_data_list,columns=columns).info(max_cols=1000))
    train = DataFrame(data=train_subset_data_list,columns=columns)
    train = train.drop('ClaimActivityID_ATGSYS',axis=1)
    train = train.fillna('None')
    return train

def get_test_data(batch_number: int, columns: list):
    cursor_1.execute(f"""
		    select 
                case when (li.checked = 'true' and ds.checked = 'claim') then 'Y' else 'N' end as IsClaimFlg, li.Checked, 
                    li.[ATG_Ref],
                    li.[ATG_LI_Ref],
                    li.[BatchNbr],
                    li.[CATEGORY_ATG],
                    li.[ClaimType],
                    li.[AP_VndNbr],
                    li.[APVndrName],
                    li.[OOB_ATG],
                    li.[ItemNbr],
                    li.[UPCNbr],
                    li.[UPCUnit],
                    li.[ItemDescription],
                    li.[ItemShipPack],
                    li.[PoNbr],
                    li.[PODate],
                    li.[ReceivingDate],
                    li.[TurnRatio_ATG],
                    li.[TurnQty_ATG],
                    li.[OrdQty],
                    li.[PdQty_ATG],
                    li.[PdGross_ATG],
                    li.[PdOI_ATG],
                    li.[PdBB_ATG],
                    li.[PdNet_ATG],
                    li.[DealNbr],
                    li.[OrdStartDate_ATG],
                    li.[OrdEndDate_ATG],
                    li.[DateStartArrival_ATG],
                    li.[DateEndArrival_ATG],
                    li.[DLAmtOI],
                    li.[DLAmtBB]--,
			    from    
                    [MOE].[prod_WeisMarkets_RecoverNow].[dbo].[ATG_DealLI_Exceptions] li
                    join  [MOE].[prod_WeisMarkets_RecoverNow].[dbo].[ATG_Deal_Summary]  ds
                    on (li.BatchNbr = ds.BatchNbr)
                    and (li.DLVendorNbr = ds.DLVendorNbr)
                    and (li.DealNbr = ds.DealNbr)
                    and (li.CATEGORY_ATG = ds.CATEGORY_ATG)
                    and (li.ClaimType = ds.ClaimType_ATG)
                where
                    li.BatchNbr=67
                    and
                    li.ClaimType = 'IN DEAL' 
                    and 
                    ds.CATEGORY_ATG = 'SAME VENDOR - AMT DEALS' 
                    and 
                    li.checked='x' 
                    and 
                    ds.checked='x'

		    union

			select top(1) 
                case when (li.checked = 'true' and ds.checked = 'claim') then 'Y' else 'N' end as IsClaimFlg, li.Checked, 
                    li.[ATG_Ref],
                    li.[ATG_LI_Ref],
                    li.[BatchNbr],
                    li.[CATEGORY_ATG],
                    li.[ClaimType],
                    li.[AP_VndNbr],
                    li.[APVndrName],
                    li.[OOB_ATG],
                    li.[ItemNbr],
                    li.[UPCNbr],
                    li.[UPCUnit],
                    li.[ItemDescription],
                    li.[ItemShipPack],
                    li.[PoNbr],
                    li.[PODate],
                    li.[ReceivingDate],
                    li.[TurnRatio_ATG],
                    li.[TurnQty_ATG],
                    li.[OrdQty],
                    li.[PdQty_ATG],
                    li.[PdGross_ATG],
                    li.[PdOI_ATG],
                    li.[PdBB_ATG],
                    li.[PdNet_ATG],
                    li.[DealNbr],
                    li.[OrdStartDate_ATG],
                    li.[OrdEndDate_ATG],
                    li.[DateStartArrival_ATG],
                    li.[DateEndArrival_ATG],
                    li.[DLAmtOI],
                    li.[DLAmtBB]--,
			    from     
                    [MOE].[prod_WeisMarkets_RecoverNow].[dbo].[ATG_DealLI_Exceptions] li
                    join  [MOE].[prod_WeisMarkets_RecoverNow].[dbo].[ATG_Deal_Summary]  ds
                    on (li.BatchNbr = ds.BatchNbr)
                    and (li.DLVendorNbr = ds.DLVendorNbr)
                    and (li.DealNbr = ds.DealNbr)
                    and (li.CATEGORY_ATG = ds.CATEGORY_ATG)
                    and (li.ClaimType = ds.ClaimType_ATG)
			where
                li.ClaimType = 'IN DEAL' 
                and 
                ds.CATEGORY_ATG = 'SAME VENDOR - AMT DEALS' 
                and 
                li.checked='true' 
                and 
                ds.checked='claim'
        """)
    test_data = cursor_1.fetchall()
    test_data_list = []
    for index in range(len(test_data)):
        test_data_list.append(list(test_data[index]))
    del test_data
    return DataFrame(data=test_data_list,columns=columns)

def compress_folder_to_zip(folder_path, output_zip_file):
    # Create a ZipFile object
    with ZipFile(output_zip_file, 'w', ZIP_DEFLATED) as zipf:
        # Walk through the folder
        for root, dirs, files in walk(folder_path):
            for file in files:
                # Create the full file path
                file_path = path.join(root, file)
                # Add file to zip, using relative path to maintain folder structure
                arcname = path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname=arcname)

old_columns: list = ['IsClaimFlg','Checked','ATG_Ref','ATG_LI_Ref','BatchNbr','CATEGORY_ATG','ClaimType','AP_VndNbr','APVndrName',
                 'OOB_ATG','ItemNbr','UPCNbr','UPCUnit','ItemDescription','ItemShipPack','PoNbr','PODate','ReceivingDate','TurnRatio_ATG',
                 'TurnQty_ATG','OrdQty','PdQty_ATG','PdGross_ATG','PdOI_ATG','PdBB_ATG','PdNet_ATG','DealNbr','OrdStartDate_ATG',
                 'OrdEndDate_ATG','DateStartArrival_ATG','DateEndArrival_ATG','DLAmtOI','DLAmtBB']

columns: list = [
        'DS_Checked','DE_Checked','ClientFamilyID_ATG','BusinessUnit','ReceiptNbr','ReceiptSfx','PONbr','PODate','ReceiptDate','POVendorNbr',
        'GrossWght','UnitOfWeight','POQty','ReceiptQty','ShortQty_ATG','VarWghtInd','ItemNbr','ItemDescr','Dept','UPCNbr',
        'ItemSize','CasePack','TurnQty_ATG','TurnUnitQty_ATG','TurnRatio_ATG','ListCostSB_ATG','ChangeDate_ATG','EndDate_ATG','NextCost_ATG','PrevCost_ATG',
        'CostDiff_ATG','IncDecFlag_ATG','CostType_ATG','AddDate_ATG','SellEffDate_ATG','LastShipDate_ATG','ATG_Cost_Ref','PdGross_ATG','PdNet_ATG','PdQty_ATG',
        'PdUnitGross_ATG','PdUnitNet_ATG','PdUnitQty_ATG','Facility','ATG_Hdr_Ref','SourceFile_ATG','ATG_Ref','ATG_ItemID','RcptFacility','APVendorNbr',
        'Merchndsr','ShipWght','ShipCube','PalltQty','QualifyAmt','RepckRatio','Status','ExcptnDate','OKDate','AdjCde1',
        'AdjQty1','AdjCde2','AdjQty2','ListCost','OI','FreeGds','WhseDisc','UpDnAmt','UpDnInd','BB',
        'LastCost','LastCost_Orig','CashDisc','FrghtAllw','FrghtAllwInd','VndrUpDn','VndrUpDnInd','PPayAdd','PrePayAddExInd','FrghtBill',
        'Bckhl','VarWght','InvQty','InvListCost','InvOI','InvFreeGds','InvWhseDisc','InvUpDnAmt','InvUpDnInd','InvBB',
        'InvLastCost','InvLastCost_Orig','InvCashDisc','InvFrghtAllw','InvFrghtAllwExInd','InvVndrUpDn','InvVndrUpDnInd','InvPPayAdd','InvPPayAddExInd','InvFrghtBill',
        'InvBckhl','InvWght','InvVarWght','AdjQty','AdjListCost','AdjOI','AdjFreeGds','AdjWhseDisc','AdjUpDnAmt','AdjBB',
        'AdjLastCost','AdjLastCost_Orig','AdjFrghtAllw','AdjVndrUpDn','AdjPPayAdd','AdjFrghtBill','AdjBckhl','AdjWght','AdjVarWght','AdjUpDnInd',
        'AdjVndrUpDnInd','Toggle','LastRcvCorrDate','Comment','FreeCs','Trans','TransQty','Trans2RcvNbr','Trans2RcvSfx','Trans2Qty',
        'TransFromRcvNbr','TransFromRcvSfx','TransFromQty','ItmFrtAllwExInd','AdjPPayAddExInd','FrghtBillExInd','InvFrghtBillExInd','AdjFrghtBillExInd','BckhlExInd','InvBckhlExInd',
        'AdjBckhlExInd','DealFlg','DealStatus','InvQualifyAmt','InvFreeCs','AdjQualifyAmt','AdjFreeCs','CsUnitFctr','InvSurchgInd','HiOldAvgCost',
        'ACChgTkn','ExcptnUsrID','OKUsrID','WDCostFlg','OICostFlg','FGCostFlg','FACostFlg','PPAddCostFlg','BBCostFlg','UDCostFlg',
        'VndUDCostFlg','FBCostFlg','BHCostFlg','FullCostFlg','DSDItmDueVndr','ATG_Ref_Orig','ItemFacility','ATG_PO_Ref','IncvOI_ATG','IncvBB_ATG',
        'FlatAmtOI_ATG','FlatAmtBB_ATG','Contact','BestOI_ATG','BestBB_ATG','UPCUnit','BalFlag_ATG','OOB_ATG','PdHdrFrtVar_ATG','MiscAdj_ATG',
        'PdOI_ATG','PdBB_ATG','PdUpDn_ATG','PdIncvOI_ATG','PdFlatOI_ATG','PdFlatBB_ATG','PdBckHl_ATG','PdPPayAdd_ATG','PdFrtBil_ATG','PdFrtAlw_ATG',
        'PdShortOI_ATG','PdShortBB_ATG','PdShortBckHl_ATG','ClaimActivityID_ATGSYS'
    ]

upcs_to_keep = [
		'8888888888888','8265798311','2073509283','5000057578','5100024817','2073509275','1780014342','2073509320','4127102769','5100019651','5000016862','2073509304','4127102564','2073509272','1780017162','2073509329','1780018499','4127102771','5000057577','1780014500',
		'1780018495','7023015346','2073509286','4127100492','1780014939','7235002027','4127102770','3663202834','5100018716','4127101771','3400031240','5100024814','1780014941','4127100517','2529360039','2073509296','3663202990','1780018444','4127100495','7023011620',
		'4900017417','1780014098','3663201964','4900017415','7023016516','4127101827','3450063202','2529300149','4127102949','5000057579','7023011720','5100024818','7023010720','3663201962','3663202943','3663202851','4127102773','3663200260','1780014937','7023016767',
		'3663200991','7023016854','2529300136','4900017418','3663201953','2073509633','4900017354','3663203576','2073509632','3663202764','3663203581','3663203249','85631200278','1480064647','1780014910','5000035022','1780015452','1780017380','3663202848','61300871511',
		'7023010744','4667501400','1780014940','4900017749','7023011716','2529300098','1780018440','5100018695','3663201987','3663203721','85631200276','2529300099','1780017989','3663203273','81162002017','3700040217','1480064609','2700039029','4667501351','7023001669',
		'74447391205','5100019650','5000019811','1300000605','4832101500','3663203255','7023016863','5000081817','8390010648','1480064642','85631200280','5100021270','7023010770','5000057203','8390010649','3663202987','1780018496','5100023314','3663207406','3663201016',
		'2529360023','7023017088','3663200210','5000042334','61300872086','2529300460','3663201039','3663203764','4470009611','4127100955','61300871780','5000045424','5400016447','4900017420','3663201015','1780017379','1480064608','3663201954','64420931131','7023016762',
		'5000010369','7023015366','7910051410','4667501350','1480031648','7648900721','5100020808','1780018456','3663201029','61300871980','5100000011','2529360027','4127101855','5000017120','4127102774','4460031888','2400030570','7023011614','3663203904','2073509412',
		'3663202850','5000017122','4127102568','5000042034','3663202763','3120023469','64420940556','1480064666','1780017994','3663202635','3800010675','2700044212','7023016802','1780017998','2529300419','5000048996','1780018447','2529300119','3663207508','3663202518',
		'4157011008','4127102253','3663203248','7023010740','4460030468','4157005621','3663203720','1300000466','4157011019','5000094779','4157005670','2529300426','5000042184','3663202600','1780017601','2500004499','7023015348','5000017123','3120020309','3663202760',
		'3663201944','5200020844','5100021959','3663203718','2529300420','64420942095','1780018446','3663203920','3663202601','3120028130','4667500079','1780017013','4667501312','4470006647','3120020007','2529300421','3120020298','2500004773','7023016858','2500004767',
		'3120022007','1600012254','7023010785','3663202036','5000042077','1600012479','1780016633','1480000581','3663203256','1780017992','3120020300','74236521685','4157015772','3663202631','5200020805','3663201027','2500010001','7023010711','5000057993','3663202602',
		'1780014922','4460031684','4138309010','3663200282','1780017991','61300873045','7127930919','4470009403','4138309036','3663203901','2529300425','1480000508','3663200975','4000055116','3400049005','3663201316','74447300033','7023017128','6414403021','3663203828',
		'2529300428','5100023299','3120026107','3663202607','3663203927','3663201009','5100003907','1780018441','1480000475','2100002649','3663201317','5000019112','74447300016','2500004491','3663200251','3663200878','2100005496','4157005618','4300000037','5000028119',
		'4470007502','74447300011','3663203646','5000042564','74236523295','7127956536','3663200834','74447300013','3400043224','4149743724','3663204256','7235002025','1480021038','3663201314','1600012506','4138309022','1480031820','1480000231','3663203922','2500013026',
		'3663202017','1600027526','3663203640','7572090006','1780014923','1780017373','74447300010','2100005535','4470002411','7127949805','5100021960','2100002632','4470009609','1780014909','4470002410','2100005534','1780014920','3120020031','3663203736','5000017064',
		'3120028131','3810011049','1480046494','3663201040','3663203753','5100018699','3663203760','4410015619','1780017603','5000058006','5100018713','2529300242','3400031248','4667501315','3663203737','3800019954','3663203919','1480032100','3663200835','5100018714',
		'1600017003','3663201318','7110000974','2500004770','2100000860','3663203902','5100018707','4138309073','4138309072','5000042364','3450063210','3663203742','2100012302','4460001656','7127949919','7023016507','3663203641','4470036035','3800035054','4470009608',
		'5000073581','2500004089','5000032852','3663204255','3663201320','5000057991','1600013972','4157005617','4667501327','2100001871','89470001014','4470000063','7127949340','7910090202','3663202633','4410019080','4127100514','4157005625','3663203735','3663203905',
		'3663203268','1600016968','1780018014','5000057546','3663201840','7127930209','7127949196','2500004949','1480021046','2100005500','1300000103','3120023012','3663203906','3663203903','3663201991','5000042444','3120021007','71279565323','7127949327','4470009605',
		'4460001628','7648900781','3663204258','3120020056','2500004087','89470001013','5200050633','7127949034','7648900712','8663178336','3800019966','2100076663','3810033048','4149743701','7007457243','82927452327','4470006662','3120020297','3663203933','2500010062',
		'2500010212','2100005495','1480031946','61300873514','7127930921','5000096212','3810012508','7110000551','4470009607','4460002002','1780014924','9668902024','4667500080','3663207384','5200020808','2529300489','3663204257','9668902014','2500004443','3000001200',
		'3120033027','1780057192','3000006119','4138309071','1780014932','7648900709','3663203733','5000017124','2620011700','3663201328','3663201013','4138309074','3800022260','5100024819','2970002141','1780017386','2500004085','4127100516','4127100515','2500005433',
		'5100020616','5000045435','81829001284','1780015210','2500005180','3800020046','3663201319','4460032191','7127956290','3800022272','3663207262','2073539287','3700040212','3120021133','7127930922','5200004333','1780019341','2073539332','4410015621','5000032179',
		'3800020075','4460032560','5100022463','7127956724','1600016875','3700087561','4470009240','3800022270','2100005533','2073539273','7127956318','2100012285','4470036147','2500004497','4300007034','2500012053','2073539306','7127949450','1780017993','7127956694',
		'81829001283','1780014929','3700041828','2100000869','5000038699','3800024296','5000017085','5000038826','4755721006','2100004024','2100005493','7127912604','2100007330','4470007505','88491211171','81829001282','5200020871','4127100512','5000057199','2073539259',
		'7127956533','4460000228','2073539290','5200004318','4129497325','1780017374','3663203956','3700040218','2100005517','63641201228','82927451250','4100000408','5000056386','3000006083','4460031221','3663204261','3700040213','88491211762','3800018176','3800035056',
		'2100000870','9668920016','3400043227','5100018697','3700075073','3663201315','3663203957','1780017838','3000001040','4410019081','5100001031','4850020274','1780018338','5000030212','7110000996','2073539242','1600016683','1480000521','64420930750','5000010365',
		'3663204263','81829001626','4470036000','88949700823','2700037242','3320002208','2100012303','2700041922','2073539276','7590000526','4149743828','1600012593','4470009205','5200050648','7626500105','3400029005','7790070615','71514150349','4460031603','7626500100',
		'3700066920','88491200234','4460030197','3800022244','1300079800','5000031058','3663203954','7127930207','5200020807','2100004968','2100005516','7127956211','1980003667','4157010993','14800000078','6414403316','3700055861','4138309070','7590000534','5100021955',
		'3800019974','2100005509','3800016046','7332100018','3800020520','3800022262','5000028225','2100005501','3600054150','4800121347','2100005532','4470009630','5100001251','89470001026','4240018897','2100005513','5100018708','889497000175','81162002140','3800018170',
		'64420979129','5000094855','4000051122','4157010964','3400021458','2970013141','7127949534','81829001485','2100006751','7127930924','3663204262','3700097305','5200012324','3700087615','2700039003','7261346159','5000030160','2400030296','4000055114','2100000861',
		'2620049614','5000029451','1780017385','1258770441','3400014058','1600048772','4600081101','5000037391','5000042044','2073509634','88491228631','3400004594','5100016887','4150080505','5100021957','2100005494','7007457234','4410015618','5200020806','5000057846',
		'3320002204','5200012935','81565200418','2700052005','2100077260','7007457231','2100005465','7066246003','2100005778','7127956210','3320097511','4460031874','82927450225','1780017017','4138309021','5100018715','4157015770','4300007327','5000029453','88949700821',
		'4460031156','4600028734','4460001594','3700035520','1113216801','88949700822','7403010285','64420930749','1600012499','3800020073','1300079770','3810015852','3010011175','4460008033','3800019937','3320002203','2100005446','4240031864','7119000600','3900000890',
		'1300000640','5100015319','5200020842','3800023152','5000057401','7790047131','2500012055','3700067002','2700041911','7910051411','3800022280','7910090237','81565200214','5000018378','2500005434','5000046772','88491218008','2100002339','5100024827','1600016243',
		'7790011553','3400001870','88949700820','4460031620','1780018919','3800020044','3700000992','64420930757','7403020300','5100019530','5000058602','71279564494','3800019903','2100065894','3600054151','3400056710','2410010684','88949729890','3700052364','81829001012',
		'1600010632','2500004484','7066203503','1258778685','5100020191','3700041829','2100007715','4240018906','5000021347','81565200234','4149743653','1600048769','3400049108','1600048794','7104002119','81829001279','4470009626','81829001281','5100024815','3120020311',
		'3800022278','88949700824','4850001833','2500004081','2550020633','7778203018','7119000602','81829001275','1480000190','3800019945','4127102127','7790047132','2100004512','5450019592','1600016702','7066201503','4450098457','2700062318','7332100025','2410044069',
		'81829001227','5000030302','5000057848','7332104417','1300079820','61300871844','2310013647','3360400419','7790019209','1780017434','4470006357','3320002141','3010010054','7104006312','7590000531','5000096209','89470001016','1600026509','7110000509','2100065897',
		'5000022530','5450019596','5410000265','7590000530','3700066932','1480031656','5100021239','3663201999','4300007103','7007463057','1480031822','3800019885','2550020629','1780017376','81565200414','5000042154','5200012934','88491200238','3760027087','3800019923',
		'7127930208','4100000362','5100023283','3400013480','7590000524','7332104406','980000771','6414403009','1600017102','9668902081','2900001002','74236526547','3400014059','3700066563','81829001699','7648906787','64420930759','2500012059','74236526425','5100003885',
		'7346137503','4470009622','4240031865','81829001181','3700052366','4129440199','74236526445','4600081151','81829001381','4300004729','5100021446','4450096683','3320002202','7007457804','61300874039','3114286468','89470001006','4000050526','81829001466','3320002206',
		'4460002030','5000029220','3700035762','2100002946','88491200237','4600028732','2900007210','3120034227','2007403000003','5100021954','5000098588','74236526405','8265771002','3320002292','2500004093','81829001484','81829001698','5000079016','7648900779','4900017414',
		'3320097354','3120027016','7346137553','5100020614','2100007129','6731200511','9668902058','1780014926','4470010295','2100061223','7790065046','7790050308','1600012399','7680800279','5200012125','7910052169','3700061079','1115605061','2310014349','5200000284',
		'74447300012','3600054165','74236526535','3700039686','2700048917','1780019009','2370005473','3700041825','89470001043','2620014061','3800035602','2620011750','1780018921','3700034085','5200000339','4470009200','2100006747','7007466855','7066203501','2900007325',
		'5100026977','7007464117','81829001424','4850002064','7007464115','64420929047','4000050547','7066246002','4149709670','3700040744','2780006570','5000042314','3400021490','5000028017','4470009604','1920077182','6414404306','4300007033','4470009625','5100024621',
		'7119000633','4410015874','2100005524','5000042944','3800020077','5000017002','5200004426','4112927463','88491212971','5000042994','2100001253','5100016775','5000042194','4138309043','4600027918','3700090816','64420941185','7104002115','2460001500','7790050311',
		'3000001004','1450003099','3120027627','1530001461','5100018693','5000051654','7590000584','3800022254','5849672300','4470009405','81829001665','7066201502','2340000201','5000057217','3000057322','1262314212','4200015716','8768401036','5000045700','5000042494',
		'3800020079','1115605081','7104000036','1450003100','3663202604','2100003045','2410044086','4850002063','7007456090','3800020528','3663204259','3663200836','5200000342','1480000188','4000050532','7648900790','3800019848','3400005811','88491200235','4470036002',
		'7471408665','1450003128','3360400212','4850020397','7110000308','3810017398','2370005482','2700037800','88491200471','88491227311','4460030039','4450098463','1370080627','4300007036','2100030165','5000004081','2100065371','4470000004','4150000031','81829001271',
		'3120027015','5100021243','3040079409','7332104405','1115605076','5200004758','1300079860','3027103407','5100024825','7471408663','2500005176','4138315450','5100001418','3800022274','6414404302','89470001019','7007457801','2100005783','1113217091','1258700020',
		'3800020037','2100012382','2100001328','7590000525','5100024812','4450098450','7066203502','74759961864','64420930756','88491200672','81829001228','7648900780','3000065970','7790065048','7648900778','5000029211','60502100208','4150080502','3663203731','1480000656',
		'2700037241','3800020298','1780018170','3400021491','81829001073','74236526415','3320097353','6414404322','4149725934','4000050530','3663202063','2100072516','1111161011','5100018806','4850002065','5100022464','3000057321','1007471408664','4410015883','5200004317',
		'7590000587','2100030047','7490836012','2100000864','3700085937','2100002679','88491212951','7104006556','2100000014','2100061984','88949715794','4149709699','5200004319','3700074759','3500045043','7590000240','89470001033','2900007324','4850020281','3980001819',
		'1113216853','1600016967','1600027515','1003114210170','5100016680','4470009621','5200004705','5000000124','4150074510','5200010245','5000059524','61300871017','3340040173','2410094058','6233876974','8768400411','1115600032','5000057442','3800012570','6414403064',
		'7332103401','4144900110','2700052409','4240018898','4800120434','2970034141','3340061280','4150088837','2100061688','4850020277','9668940016','4149709694','7490836011','4900017421','7066208273','2310013529','3077202430','4780000053','3800026995','89470001002',
		'1113214544','1003507448529','3663203738','5000092910','3340060108','2500004099','3663202853','1600010371','3700044311','4800121357','4200094475','3663207190','2620014454','1600047864','81829001277','81829001280','3010010308','5100024619','3800035902','4450098451',
		'7490836062','4460001593','1003507448530','1114110040','2700041926','2529300458','2100000905','1113216808','4600047985','5400054245','7490832438','5849672306','2073509642','5000066491','1480000440','4200035501','7332103366','3400040211','3800022116','4300004211',
		'2310014342','2700041924','3000057328','8768400399','65762211176','89470001005','4300007035','6414455556','1300000978','1300099390','3700066575','89470001001','2100001588','5000057281','81829001382','3900060020','3700000995','3800022256','81829001857','3400021464',
		'2100001641','3700044800','1862770316','3400044000','1600017761','5200050632','81829001273','1780017371','2100002631','1111161168','2073509635','3100010101','30521032472','3700039363','3120022821','2200063140','5000096210','3400044709','2100007188','3700028616',
		'1003507448531','2100015135','3800024473','2529300699','81829001692','1370096725','4850001830','1480000210','2100012164','3800012572','4850020291','7007462605','81829001463','5150072001','2310013648','5100020244','5100016721','4470009623','1707713264','89507200039',
		'4450096680','5000042874','3077202429','1600015940','7258670130','1707713263','7127949311','2410078936','5000017003','5000016928','4300095117','3700074651','2570000320','3810016963','2410011659','5000042974','1003507448527','3700092810','3900001893','2620046170',
		'5100023292','7104002133','1740011811','1450003130','3700066557','3320002140','2100000903','2100006752','5170078826','5000090326','3663200839','3340061283','2700044205','3040079425','5100010816','5100019759','4850001829','1114110528','4200015290','4800126617',
		'3760027213','5100015007','1480031825','3700098216','7790047140','3400001871','1114110270','1600017082','2340035001','3663203866','5200001361','3800020040','5000050157','2400024115','4470003330','2100060269','4850000717','5100018706','2400062251','1480021089',
		'4100000413','1600012610','3000001321','7680800283','5100020847','8768400401','1370001723','1920079556','3120000066','3340040176','2700041904','4300095114','1707713267','88949752479','5000029336','3700021461','3500045118','4100058697','3114235902','3114252385',
		'9003114204210','3114235900','2310011627','5200012936','2500005677','81529400014','4200015421','7342000125','7007464172','3340061252','1780001278','8259201074','89470001032','3800020530','5100024625','1920078626','7342001614','7648900768','7119000634','4460030614',
		'3000006322','1003114200126','81829001180','75703795068','4150084737','7790049184','1780014915','7261346144','7047040389','3800022250','3340060120','3663200104','5100001261','1600017104','2113190552','8768400408','3663201864','7104006331','81829001179','3800025865',
		'3940001647','3663203860','4240031863','74759961868','4300004574','2410078894','4178847215','3800016770','64420941103','4600047972','100311435899','2073509639','88810911004','3010047324','7590000220','3340060109','2370001625','1003114235867','1480031821','5000058090',
		'3700096257','3600054161','5000057815','1780018045','6414432170','4060037908','1630016574','89470001004','5000029264','2100005192','3000006354','2220095036','980000773','7756700158','4850020275','82927452323','7835557010','3700056804','2310010755','5440000008',
		'1003507448536','74759961866','1600015980','81829001259','3800027010','1370082425','1980070195','2500001069','1370083812','4335424258','5000097514','5100002379','2500004055','3114200069','2840067285','4410019074','1600027549','5400012219','1600015163','2550081121',
		'7332100027','1600017756','5215900002','7342052440','7342052420','1980003674','5100014292','3114200034','5215900001','3700097604','3760010862','3600035970','7342011214','3114200472','5215909004','1313006053','74759961867','3600038586','89470001010','3114200525',
		'3700075264','4150000025','2100000904','1003507448533','2100006405','7023016960','7265545442','3114200011','1115605926','3663200108','1003507417721','3663201838','2410044076','7007457263','3100018454','3360400497','2700050015','82927451912','7119000817','7047000307',
		'1003114200485','4850020272','3700077251','3800019870','5000058211','2100005317','3600041926','1600027567','2410093995','5100021248','5100021450','4133300232','3700061071','1780017528','4410019075','4410015766','5215970115','4450020148','7332100005','3940001614',
		'1450003098','74447300034','3732311477','3010010056','3800016777','1450003183','2100003256','4300004202','4850020295','3114200036','1480021078','2484275121','2340035000','5100024616','63641202128','2700041901','2100000728','1003114283218','3760028437','8259201073',
		'1630016820','83609391002','4600028735','4154881182','8259201071','1630016576','1313000612','7047040391','7910053186','5100020807','1258778564','1530001463','81529400013','2410078886','7229010224','4850025675','4300004573','3800022264','81829001740','3114200553',
		'3663200112','7342000128','3663203928','3114255405','1003507417741','2740047017','2100000730','954202607','89470001012','7047000300','2100000925','2400049781','3010010067','2410011508','5000017227','7119001601','5000050425','2900001608','5100014293','4144930013',
		'2500005676','5150025516','7119043672','1003114253415','7007467725','5100021445','60502100288','1780016541','7342051626','1600016361','1600027532','2840009215','7047000109','64420941065','7258670100','4200015257','5100026844','4144911140','1780017372','2700000266',
		'7104000034','2370001415','4850020273','82927452326','8201110228','4450098469','3400004593','5410000400','4300004568','1254601110','4200015969','7192153484','81829001285','7047000129','4100000324','3700000998','2840001537','2900007345','3700011045','1003114235869',
		'1740014000','7104000060','4300008803','1780018129','2113100039','3040079398','4133309161','3700097584','7756725423','2880059995','3600053743','3340040171','3700060821','5215900520','7332103481','2420005016','7585600110','1480000285','19600524250','2410059474',
		'1980003663','3700087549','3810017121','1081829001147','4100002278','3800022268','2400024113','3500045044','3900001894','4178847115','2410059440','2370005483','5100016777','5000042324','5000058234','5210009860','5410001160','3600054167','3810016966','2113100021',
		'4149708664','5000057444','2100007713','4133377364','1740011851','6414404315','4850020472','1600015840','2310013533','2310010756','88133400046','7332100023','1003114200183','4149708663','1920079329','7342051646','1920098016','2500004083','1299340106','4300004728',
		'81829001385','2700040000','2700041900','7007400240','7047000323','81829001739','81829001725','86170300011','2100002322','3320000183','3114265402','4300004608','4850001775','7910077273','87744800162','5170075713','5200004314','5000043792','7066223003','87744800361',
		'1600015136','6233800230','6233806130','3800027002','87744800479','5200004748','7308090759','7192188222','3400007053','3400039997','3760037877','3663200111','3810015863','4240006284','3010026351','81829001467','5000051278','7261373945','87744800364','3400024100',
		'4223830220','7047000302','3040022244','7047049646','88810911007','7265500115','5100024626','2900007650','4300004209','3800022246','2100061161','5100006799','7336040130','1002480062002','3600047357','88491237927','5000042504','81529400008','7110020005','3000045017',
		'5100005919','4850020368','4600081215','5000010085','86170300010','7066208279','613008735142','7047000313','5000010046','6731200550','5200004154','7258670110','81590902048','3700066715','74759960721','1780018022','3120023463','2800050346','3120020157','7680828008',
		'3340040172','1370082235','7572071003','1254601108','7790050212','3600049413','4850020276','3010000133','8265750800','4850020271','4154867965','3760000642','5100015876','3600034104','5849672303','87744800360','1600016344','7047017535','2073511001','3600031803',
		'2700041902','64420941135','2840058671','7047016473','5200001359','4300005084','1600016684','88491229799','7007466546','1113217115','5200004457','1600018403','1113217139','5100015874','3700077810','5100024615','87748003686','2100060085','1262314209','7007457266',
		'7047018744','88810911001','1630016821','3400072062','4300008182','88491237928','3100018464','1600012025','1254601106','81829001425','2100000872','8768400410','4850002238','89762900002','4427605546','3400043001','4600028869','2100007327','63221000532','4470008693',
		'4850020293','7047018495','7342000024','3600048750','2840067284','4300004726','5215970114','7756700156','5000029452','3400029105','7336040132','7184009203','8259201072','3340005553','3600047766','3700061712','4060038023','4470008713','5100025031','1450001269',
		'954202608','7308080419','7047027831','3400020350','3100010102','81590902037','9396600586','2100007164','7066219661','3980011009','2100005189','7680828009','6414403031','88810911009','1003507400337','7047013767','4460060105','7835431152','1740010077','5000042984',
		'5100016888','4100002286','2200022037','1600015950','1740014055','8265750561','4850025012','81829001196','2620046770','6414428260','4133309061','7572000599','7047000319','7336040125','7047028447','81829001177','3810015861','2400051235','2073511006','5000042914',
		'2410078896','954202823','5000029330','7648941464','3000065300','3000057183','1600027855','7910083050','71516604402','3600053611','7490836021','5100016681','4144930270','65827620251','9955508003','4335424252','4133300057','1600016346','7403066167','81829001693',
		'3900001892','2800094873','7173000716','7835557000','2400052587','4450096674','71516604405','7110000550','3700058651','5100019624','5000042904','2100061218','1780015845','2113190554','7835430355','1630016824','7790050213','1600026508','7066212603','1299340120',
		'7119000805','1920075352','5100010662','4200044363','1009955508533','81529400007','2700048195','1600012185','2007403000004','1530001462','4900009362','4144960123','2484270257','4460000938','7265545441','7265500117','5450026019','2113100053','7007455957','9396600689',
		'1630016906','1299340108','87744800362','7342000015','2100061243','3663203732','1003114253492','3340040135','8265750573','5100020249','4000051125','1600015990','7265545445','3700055867','7342000011','3700019508','7261373943','3114200029','4200015259','85706500702',
		'5100021242','1003601673800','4180021100','88949779564','1003114235975','5215970337','2700041921','1630016908','4200035502','2410011440','3800020069','2100000525','2100005193','1600016710','5100003920','4144930022','4178801435','4300005789','5100021498','1299340121',
		'4300005106','3340080172','1070002440','88491200236','3700044105','63221000320','7299944123','7265500110','81590902045','3000032122','3600044561','2100000906','6414428250','4850020289','71516610401','4850020367','7007467165','7346122553','4149725935','3700096595',
		'5000058005','4470036001','1081829001934','4470008690','4060038021','87744800357','7110000604','3100010103','87744800716','7910052072','2200021737','7092047636','3114200043','1003601673801','3700097587','81829001657','3100012612','87744800354','3400072060','1600049438',
		'1960004610','5100022212','7265540509','4178847710','2113190555','7047000303','1003601673739','3400024000','5827620253','4600027342','3500014476','3400005201','7626540418','3980012990','4200094444','4427605594','2340005728','3087744800751','7066246001','2310014363',
		'2700037837','8259263124','2113130267','2113135049','3900000891','5000010088','2780006282','3760014051','1003601673802','5200004328','5071516613802','7092047633','81829001175','7490836064','5400010060','5000029257','1002480062005','7590000617','5215970103','6414404292',
		'6731200551','5200004232','1003507400492','5000042954','5215970119','3485622698','1480000622','1003507448541','76172005820','1600015128','3000016947','7229012240','3746603062','2880029100','3700039300','3700040567','87744800369','7047049647','7790011633','5000004061',
		'1002480062006','3700096256','1300079890','3400000310','3700095688','87744800766','3120022025','3700077179','2970013166','2970000138','3800035702','2780006790','1113216658','2700048196','7299926203','7047000310','1600016849','7192100338','3800013840','3400043095',
		'2970002147','6414432286','82927452325','7680800281','2100001086','1312001251','7007458058','7403010300','3700076622','74236500270','7910082239','81829001156','7047018649','81590902103','5000017001','3400020200','5100018691','3800027660','60069900154','3700087566',
		'2700041982','4460032419','4850001776','1600042070','4850020348','2100060270','3700008265','1003507448543','3700097783','7007458049','1780017416','2100062679','5215901330','3100067090','7490832441','1600016389','7910083102','4460032563','7357013009','89470001009',
		'7346122503','3400000320','7265500101','3100000730','7835555000','1480000209','3340040125','7047040387','2570000310','5000040241','7910094203','2200063144','85407400602','3600054281','3700097307','4600028876','1780017470','4300000052','715166052014','6731200549',
		'1600027488','8259263123','4800135366','4144900186','3700090243','5210009690','7047016592','5100023355','7680851829','87744800355','4460030024','1630016822','7835470047','60502100261','5100021240','7336040156','2840067283','65827620255','5100020245','4460030924',
		'1070055686','1002480062001','3120034327','3760027094','2073513631','5400049668','7047016324','3800013842','2200022035','1780018012','3746602979','2700037831','4000055085','3700060567','3120023466','1740010204','2310013531','5000058069','4149709780','5200004706',
		'1370081430','5000060821','4850020184','3114267040','2200022036','88810911385','4460032260','5100024613','9955508060','81829001791','4300005670','9955508588','7047014783','3905955112','4149709735','2073542095','4149709751','88491229801','9955508792','3340040151',
		'7265500105','3100019601','1450000256','3400022100','1299340111','7047016594','7756725425','5150025362','3810016968','5100027357','7835431909','3600013792','4900018026','7680800683','2113150663','76172098749','4460060106','5000058604','87744800356','7790019206',
		'3700040690','5200013517','4300004607','85407400606','9955508922','1780017828','85331100371','3600033592','3600054266','2100000901','1600017108','3700074969','1920081700','3100030744','3100012020','5150005722','7047000321','5100020195','1480000189','4850001779',
		'5410092502','5100027358','7047018743','4112907762','4119691088','7585601221','7630618077','85407400604','3340060119','63221002699','4200015804','3000065940','2410011486','4460030112','63221000099','2113150662','3100000736','7110000443','7255484811','1740014031',
		'8768400396','7790019416','2000011197','5200004231','2460001470','4850020183','7457011400','9990053105','3700040152','4460060111','7357053045','85706500701','2200021483','7835470759','74941788310','7835558000','2700041909','3100012023','2100005228','2100062316',
		'7910085174','7403081822','3400038620','7047016593','1780017058','2370001411','5100021886','5160000001','2073511022','7357013000','1920079326','1780017375','1780015130','1780017369','1370096735','1299340112','3600047622','6233898552','2100004969','2570000382',
		'6414410625','3340005570','2370001413','3800026993','74759964012','4900007692','4144930011','5100024816','5100024820','1600048764','4119689107','930000086','3700077131','5410000562','1480000370','2113190532','3800022284','3700099042','7066208271','7192149989',
		'1630016825','5200001087','2073511016','2310014154','65827620250','7192185863','7357053048','7047000643','88810915004','81829001666','88810915002','2073511004','2100061689','85331100376','7271400222','3120000058','75703795086','2113100040','4112907712','2880099828',
		'3700025787','4440013828','7119042189','1070080723','5400010183','7047049649','3400017041','2740047073','4600082111','3732311441','3800022248','3010012736','75703751525','5020050000','5000017604','7265500102','7835470843','4129497326','74447391235','3700075139',
		'3760035491','3340040123','4000042206','3340061282','3700008622','7756700154','5215970125','2410011665','3000006441','2780006313','87744800679','5100017966','3000031678','3700034899','2100000727','5200010241','7357053047','3400046061','2370005461','5210059987',
		'3663201106','3500045038','2400052586','3700019096','7265500107','2410059472','5440000004','3500045830','4150084090','7490832439','7490836013','5071516613803','5000010387','2200022321','3700093138','64243412204','4600027341','2100000521','2780006477','8259263120',
		'6731200552','7192172246','88491229867','85407400603','3760090570','7680804957','1600016961','3700087455','85961000009','74759962273','4149707495','74236500434','3507400557','2100000923','1300079830','5100006959','7047010376','1780012595','7047049654','7119000341',
		'5000032193','2700041916','7261345171','7047014171','7007464307','3746604700','2700041920','3400047056','1003507400543','4149707508','7066242401','2400016302','3340040178','3700041460','8259263126','1920095871','4300095373','3800013856','1980000117','4119689102',
		'81829001851','4300008184','1740011000','2484232111','4300004421','2310014359','954202821','5100027013','7339000731','2840058670','5450019593','4850020278','81829001805','4133303961','1780014312','4470010301','1600017109','7047016497','5000091966','64243489704',
		'3600033593','1708288525','5215909003','4149709747','2000010473','2700039914','3340040143','22200951029','5000096311','4300004721','3100012610','4149707506','6731200564','2200021484','5410000432','4850001782','3700077811','2370005468','5100009503','4133309361',
		'5100019625','3000032123','2370001493','3800024565','4430012513','3400022000','2113190553','8434812785','3700014392','5000050136','3500045041','2880029445','3760020197','4100000301','2113150085','4850020292','2200063142','2484275126','5200001088','3760048675',
		'5000057409','4149707507','3800022266','4133304061','1380055132','5849670122','4132100975','3800033945','7119000698','1312006082','74759961175','1700012247','3400021463','5000050427','4850000818','85331100374','3000056204','7110000547','64243489701','3700095585',
		'4133319235','5215970107','7192162849','30521008600','2310014380','3100012018','2529300171','3600047199','3087744800866','5000054431','7274500865','3000056728','4149707480','88491235916','5100015251','3746604726','8259201108','3120022785','4300004658','2830000433',
		'85407400601','2310010397','88491235915','2830000405','1630016922','6414404164','7007466899','7490836063','2830000404','2073511002','7495618057','3663203907','7299944223','1113217087','4180020850','8434822726','85331100395','60502100209','6233877940','3000004160',
		'1920000601','3700028193','1370081240','3040022256','4300004204','7192160132','1009955508532','5200004759','2570000381','2970013162','2310014001','1780018028','2484275122','3320002218','7119000657','78656055727','3100010153','1600013463','5100015245','2410094026',
		'5000050432','2900001607','83609391003','3400024600','7192170100','72277621010','3100012016','2970013138','2100061541','85997700526','3700086234','1740010177','7332148465','1300000464','5210009150','5000057919','4200014445','3600053637','7047016487','4460030548',
		'7648900723','3400008011','1780018013','2200027971','7680800282','1003507462237','60502100299','7007453983','4100002268','2113150675','1920080306','3940001973','4000052539','5000072501','3940001635','1004082201114','1299340114','2400049782','4460030438','4470007987',
		'4300006089','5100004312','1740010076','2100003405','1600049437','7680850106','1002480048525','3800026974','7457065057','2100060485','3400000330','4300004200','4850020401','85997700512','1920081143','7066208270','4460032347','3338367602','1600027489','89507200043',
		'1003114285277','2073509641','2100072903','2310014357','1740014040','2400024656','3500014177','3800016772','7271400240','3700085936','7047018741','3338367601','7590000230','5100021230','7403010195','3600053746','3700097589','2700038032','8434897735','2400016286',
		'2260090125','7357053046','60069900165','2310000073','3077203510','1600012222','7048101163','930000180','3700086224','74941788747','81529400019','3545777201','2113150111','3000065070','7047017404','7000212996','3700086225','2700041905','81829001850','5170020613',
		'5100011285','4470010320','2073511021','1650010001','7835470770','67852304006','5100021165','2700050011','4149704019','7910051446','7490832429','5100012804','3800027007','2113100022','5000042964','7047016485','9955508663','7750700005','3700010776','7756760300',
		'4180022900','3600039299','7418245992','7790019282','7750700003','3700074984','2700041903','4149709781','4000042431','2100007189','5480042338','2310014343','6401445805','2740046537','1009955508999','7447100634','7020050439','1003507400821','74941788730','64243489705',
		'954202822','1600048796','3980009093','5215970127','7274508936','3338365020','4150084560','1003601673804','3663207703','81829001828','3980009094','5100027974','4850020398','7127956099','7750700006','4470006441','2370005485','63221003814','7790050318','2620011730',
		'2100061987','9955515154','3600050108','3890004271','75703751523','2073542093','74759960725','4149709731','2500013023','5100017101','1004082201154','1980000128','2800094886','4200015738','4600028736','5020050002','4149709782','7066212602','1450001933','3700068232',
		'4149709745','7240000726','7047014212','2410010442','63810267774','7020050420','1003601673805','8259263195','5100015875','74759941420','2100007231','2100030028','7047013792','3700047971','1960005030','3746608323','7106804126','65762211184','3700099929','5100000020',
		'2100000523','7184020500','2100006068','7192125885','7680850738','1009955508832','5200004400','4300095350','3800033947','1570005144','2100007191','1003507400542','6731200525','3800012571','2073511014','6401420805','3600053744','76172098840','1600040670','7265500104',
		'7265500106','1800028794','64243489720','2460001499','1780017414','8265754221','3338367573','3010033292','1600017757','2113100026','3030000102','3010078457','3000056731','1700002653','4149709746','2700037943','7756700121','1780016974','4460032417','4149709787',
		'3400001875','2620049615','6414410626','7047018746','9451442915','2700037912','60265217282','2410078924','4156514124','1780017678','1312006286','4100058832','4200015397','5150001228','1114110027','3700076470','5000047252','5100024637','4200087459','3010011201',
		'1450002131','5100017961','7835471647','5450019597','7910083625','3400011470','2073511007','3600048756','3890004207','2100065320','6401429352','5000018300','5069511912000','2200021735','4149709760','60265217250','1600014732','3800016249','1600049038','2200021488',
		'64243489703','4100002246','1600017763','78656055732','7648900703','3400017716','1740011806','64243489721','2073542094','4330130581','1600014709','5000029262','1009955508531','1920002699','9955508050','2113100114','3040022258','1780018021','2200027978','4112907790',
		'1003114286531','1780019084','5215970473','3700097306','2400025080','64243489702','64243412201','3700095583','2620014474','3980009201','81590902047','5160000004','3120000061','3000057323','81326702007','1600016939','60069900328','3700078061','3100011305','3800020519',
		'76352800008','7047049653','74236560710','5708201000','3663203859','7020050444','3890002970','1258778900','5000050431','1530001495','5150024177','3800011896','3760035495','3700087453','7835471648','1600015762','5100010656','4149709914','1600043054','2700041919',
		'2100000731','7457002400','3100080921','3077200206','4300001049','2100007762','7066219662','1003601673736','1780010031','2100004583','6827434224','64243489712','3900008663','4850020129','7790050207','5100020472','5000044746','7680800419','6827491160','4200094430',
		'6827434216','5100015241','5150000679','3900001567','81326702000','3663207267','2113190531','4240031861','7192194675','81829001829','7274508829','81829001831','3485691088','2800096253','3700057071','3760023973','7910090212','7119001361','2200027611','7457095015',
		'7680800388','3400072335','3000001190','3010012734','3400048000','3100000735','2410011061','3338365162','3760023200','4300004208','7232012442','5000038771','2400024657','3000016900','6414432287','3700076554','2100005238','3700096854','2100005718','3400031800',
		'7007457269','3700086235','4850002237','2113190533','5215970471','4470009407','2200027612','1780019395','3400024500','4300006090','64210272777','3800035502','89762900003','3810017609','3940001977','64420941285','3010050657','7910088251','81590902044','88133400047',
		'3890072061','3100012019','9955508527','3800016248','4650021613','3700025574','3800019889','3800035003','1960004630','1920000706','1901461090','7418246325','3700076621','2200022026','5200004708','2970002146','3800013862','1980003661','1400000630','3800017867',
		'88133400049','2200021486','7047016496','1740014001','3500014172','3732311582','7418246323','2073511010','1980040109','5000029260','2100073343','74941788341','3980013687','1114110136','7648900638','1780010032','3700078893','2100000522','4100058838','74941787641',
		'3700096250','4154841928','1900008504','2570000350','6827491161','3000057242','2570071147','5100024623','1114110170','3940001974','5000004071','3340060179','64243463101','3760010533','2113130150','3400031840','4138315548','2410000000','5000075682','3700076549',
		'3000056828','74759964015','5000096361','2113150475','3100000722','7457009400','83609391004','82927452321','7127956139','4300008218','3700078063','60265217286','7910077255','7020052452','5020050003','3700055914','3700068235','5210000626','7910083630','1005450019145',
		'3010047357','64420940489','8768400409','1740010235','7274506836','4112907764','65724317512','4116705350','7910088252','4149709788','1480021080','7110021090','4100002266','5000015991','7066246010','63221000701','4300004203','3700086223','3700096260','3087744800863',
		'5100024617','3400017516','4300008219','7418247352','7274508619','5440000296','7265545457','3338369999','5100021449','4430012172','9007274500782','7910014947','5400054258','4427605612','2073511018','81590902062','74941788723','3940001970','4780000015','19600570833',
		'88491235619','2113100063','5100018035','7047018502','74941788297','3980013126','5210076249','7572053228','3150640202','1920078630','3890003011','3940001608','3800021052','7572000555','2550020226','7020050425','7020050421','4149709741','6827434599','7110000438',
		'3100011119','4460031220','2400024116','7066219663','5069511912030','65762241984','5100023345','2170650006','7630172212','3340005559','3940001594','7127956516','7092047634','4200015811','2370001412','5150000048','2310010258','7910076159','3600047926','4600043151',
		'7047018388','1370034263','7274500855','64243489709','5215970124','3400014061','5849628128','5100013809','3700076971','3000056734','72225296188','3150658701','1450002419','5200013518','1780019480','1740014004','2410011923','4100000287','4149709759','74941788099',
		'7192127624','2100061146','4460031396','4100002290','7910052023','4149707401','3800019858','1780018805','7119001697','7630618076','2100002693','1450003134','7910027809','3760035031','7299924903','1600016893','88491200680','88949784125','7127956514','7047016486',
		'3700085935','3700039346','1003601673738','61300871585','3340040124','4460032387','88133400051','5000061490','1070006147','2073511024','3600054316','2880014610','6233899062','3800025394','4154888775','5000049390','5200004403','74941786057','2970013147','1113216875',
		'5000017191','5000057530','1312000828','3700076533','2880029441','1090000018','69511912013','7680800655','1450001796','954202825','7119000808','3980012994','4144947154','64243463104','2100073345','74941156546','5100010658','9955515170','3700079129','3400001874',
		'9955508516','5000050138','4100059299','4100058837','3338367531','1600043101','2100002650','2340099809','7431202610','1833701212','5000057413','88133400279','7418246329','2500010149','3100080943','7418247369','5150000686','7910002860','7100740671','4667502620',
		'5000045683','3010012732','5215970413','1004082201124','2310010155','2100065356','3338320027','3010048717','7271403969','2580005240','3663203857','3700085934','81162002128','1370032294','7790011563','7740012853','1780017368','4154858741','1300000112','7480601500',
		'5100027009','7261345514','9955508061','7457000400','1600049041','5400036371','4144940252','3400099982','4149707382','1800000260','7261346164','7127956517','4300095407','9955508650','3000057340','5200004234','5100016461','3400040040','6233875551','7271407047',
		'4200035237','4100059362','3800019933','88133400076','1299340104','2113130135','7110021079','7040900406','5100018648','5100016150','3700035669','980000777','4330137000','5100024890','3700084981','3500053041','1650010003','4300004648','4300005671','88810911006',
		'1114110290','1780014315','1480000283','1800052230','7910076158','3663203858','7047049652','83997784889','3100012611','6731200526','4180050150','7007464301','2840058672','1600017759','7626540412','2700041983','4100002263','2100004434','3000057050','1530020045',
		'1500004830','2880029743','2420004784','4460003191','64420941175','84024312258','3940001985','3600054305','4300008183','1600015233','3700013093','7274506369','1600010708','1960004580','4300001660','7684000324','3400017315','3700076553','5100025301','1370021674',
		'4300008216','3400025812','7106801126','1630016823','74759961274','7457081011','1600026460','1004082201749','5000050254','1004082201750','5000050429','4100058698','2410010444','3700079604','5000029425','4600013061','1530001448','3746602826','2220095100','7100701868',
		'1600016254','4780000021','69511912021','5000050428','2200022323','1740010078','3600053742','7192150365','3980009090','7142909535','5100021233','4670472090','3338314605','7835470046','2310010517','74941788358','2200028008','7835470821','63810267507','1700007191',
		'3800018366','7418247338','5400011971','88491228176','7265500114','7457065176','4600036358','3700076469','7740022640','2410011490','3100019677','1600046684','7457027400','2400049783','9955508300','60265217284','3800016253','2400051085','4300005388','930009531',
		'4470008689','7008503508','7339008436','3700090802','3700076416','65762211185','4100058793','2113100064','5100023404','3500014178','3700097799','1600014888','7127956513','74941788082','4300095369','1299340113','4180020223','7066242402','5100018064','7740022641',
		'7740012733','7104002135','4850001866','3940001634','954202824','61124736757','7680800685','1700013829','7265522011','1960004620','2780006394','7774529188','7255461989','980089525','7774529186','2073511017','81829001826','1312006258','3100000738','930000650',
		'3700087076','5150001700','7092047659','11200200015','72225219170','7342060122','7740012713','7342060121','7740022637','7127956515','74941788105','4100002253','5100021225','4850020131','3400015251','5410001870','4132100645','3800019109','5069511912015','5200004369',
		'7271400646','3100018462','2100000945','1009955508521','3800013879','2100004919','3890003033','3810015864','5100024868','61124739033','7100714937','4144930197','6343589151','7750700008','7648900947','7265540581','88133401280','2100002633','3760018728','63221003732',
		'3400056065','4850020396','3732312527','1700013869','88949787776','2800032002','5100024620','1600016367','4118810192','4300095353','1114110275','2800030396','2400016301','5000042794','2700000933','7457004400','3700074968','9955538663','1920080833','3600053745',
		'1450002420','4156500027','1740010041','5150000696','3010050655','45425165569','11200200016','5480042336','1920098010','2073562001','7457001400','7774529185','7680800276','76808008791','1920089333','3600051516','1003507401220','1312001417','7756722600','2484271125',
		'1007274506370','5000069270','7173000720','7192124858','4100058795','7119000672','7339000730','4850000100','3746608324','64034404214','69511912014','3700027661','67852304010','3600040531','7127956510','7255428800','7274500858','3890003008','7127956544','5150024351',
		'5100023284','3700076510','5100027477','7342060123','4150100923','2220095101','7045900573','2100073299','5150024234','5200013515','1630016921','1450001809','1312000040','2310032784','6414404748','5100024614','7110021015','7265500212','1780015576','3340040136',
		'88462310164','7265501119','1780016984','3000056208','7680828017','4800135450','1450002268','7047018512','3000016910','1450001973','83609391061','3340040175','1600015781','7774529187','5150024357','1920002569','5210064958','2073511230','7007466544','2410011470',
		'83609391128','4133319435','2500003565','2420005020','3120022876','7910052320','4112939640','8768400400','2113100041','3400072123','1800052340','3800020521','2780006778','7471408682','2420003835','7100714842','7835470727','7418245993','3600054277','2200021734',
		'1111121809','1800000185','7940026540','4460032263','3700085926','5410001290','1740010245','1122564368','980000293','8259263135','1530001478','3700077129','2200025584','69899780913','1450002345','1450001823','72225266506','5100023315','4200016415','3600053332',
		'5100020611','3100080922','3620000250','3485691082','1480000620','4100058835','5000039818','7756772162','2340006585','2100002687','7418247351','7457065030','5000050365','88133401102','7127949917','2700049018','19600525188','3800021228','2073511003','3600050467',
		'1258778366','1450001995','1920078914','60265224411','2410011663','1600049419','7066242403','3400038645','4154846335','5480042333','2410011913','2410011059','1122564369','2780006555','81590902084','7790019257','2900001669','1780012860','3600049679','1003114253488',
		'3040079265','1450001784','4610000123','8259263134','4460031387','1450001964','2780006557','5000016924','1920074186','2580002376','5480042339','1920080834','3270015468','3600045128','7092047656','3500051088','2548400657','6220080995','8133400298','3700004451',
		'83609391077','5150000675','2370005474','3663207268','4149707394','1600045723','2113150645','7585672871','1920098012','85331100375','7835471700','3000057243','2310000158','7046243124','3700084539','74941788716','2819000729','89505900061','2840060939','5000077537',
		'3940001605','7910011577','72225263330','2970002148','3700077267','7910075992','1800081778','3890003535','2370001405','4200087574','1707713282','1780010974','5100006818','3400031850','1920098011','3800016250','1780017411','3600047932','60069900093','8201110234',
		'1116211104','5000057932','7127956541','3700096254','5849672301','2113190534','3760011444','1800000211','1300079850','4157003898','3500097160','2200021736','2410000018','3800022067','1003601673737','3076803121','3700078059','64786596299','4100059297','3100067087',
		'7192167598','7218056666','7066223001','4300004610','3760048279','7740022638','88949789442','3040022091','7910052772','1007274506328','7045900930','980000781','4670462010','61124735071','6414432300','4129491029','45425167013','1530001479','4450033915','3800026997',
		'5410001310','3010011962','9990071432','2310008541','7255426037','7007458585','4670462020','7100740505','1530001498','4000049751','2200029044','61124737113','3340060198','4470010319','1740010951','2840001553','1480000743','83609391071','2073511263','3120000159',
		'3500097161','7910000848','2113190539','61300871584','5000050068','7457061007','5200004702','1370024527','1002480062003','3400024700','1800081773','8259263125','7750700004','3500051093','5100000007','1312000291','4470009216','4670479612','7192125316','5000096303',
		'5100028101','3600051358','2100005242','3700052382','3000032120','1003507448532','2400016299','3700075488','1003210005529','1700009289','4000044327','2310014340','1114110335','85331100372','7192175483','2100065321','4132100715','2700037884','7255411119','4850002123',
		'72225296028','7457008400','1600030650','1800000918','7667705720','2970002131','7457004408','1707713281','76352800022','7585602415','1780012862','1312000133','4600028877','3620000440','1300079840','4133533425','3600034103','7007458055','3270011053','7457003400',
		'4330161164','4149707523','4112907702','7047018507','7740022642','1450000301','4157005725','2200027973','2200025586','1700002404','61124738703','5150000683','1003210005527','2260095302','5200004678','1480000082','3500051406','3800016513','1114110023','7232011412',
		'7342000025','1600014705','88491228433','2370001864','3100000737','7457053400','3700041653','4141942005','7910051022','81529400022','1700009274','60265217211','5100026843','3600048124','3900001562','7218056665','1707713274','3500014170','7240000727','4800135428',
		'4530000549','3600049061','2100004875','3980010803','7218063810','3663207685','2100060464','930000044','64243412209','7910078555','4600045062','81829001830','4240031862','3700082881','1600046679','4200044213','3000056729','1600049436','3700061111','3746604269',
		'2100005447','5100003813','3760030074','5210056811','5100019570','1004149713395','1070002152','3500044673','3760036491','7255490362','3760001786','930000080','5480042337','1600018952','74236500318','9990071310','3700071507','2400052374','7910052006','3700075102',
		'4149709790','4300008181','7240000725','3270011057','7258680600','2550030407','4000042208','7756725426','5100009502','2310014366','73052152103','7457095003','3600043586','5210038245','3600051517','9955538520','4000052537','4300008180','4132100655','2073516124',
		'3890002919','3700076161','3000031175','3114285280','2100072901','4133303561','3700078429','5000050134','5000057580','4129491011','1450001992','3010012006','7100714576','5200004327','1312001261','6401433352','7490832428','3700095681','3663202727','2100000718',
		'2700050010','1800000210','1114110024','7040900404','3700080688','7040400100','1980070251','5000050070','7064002169','3340040128','4950812600','4132100545','3700078057','2400055454','2113150802','7066223002','3270015469','1740011835','5000017165','4800135472',
		'82927400622','3732311611','3800023433','85961000003','3890003032','1450002279','3400008752','1004149713390','7271400663','7255461421','3100067106','7910077272','3700097810','88949700108','1600016348','954203169','3890000947','3400029605','1258779008','2200022278',
		'1800000182','3600050979','82927450572','2100002678','7299980313','3077203222','2113100062','7127956511','3760030323','1450002421','7680800418','2800024584','3400072061','1920083721','19600524234','4330161187','1450001416','3100080947','7910051479','4240028967',
		'7046208376','3700091064','1370009130','74759961065','2420009421','71575610001','3500051087','2073516166','3150662402','1600014734','3270011054','3100067104','4300004727','1114110002','1450001968','1800000415','2780006582','3600051357','1312000026','2310013983',
		'7265501121','7910051021','3746602980','2570071143','3890002900','1530001468','2200022585','2073516173','3000056855','3700033836','72225216069','2400051115','2800032007','1450002415','930000334','4154873705','1480000742','3000065040','4330161171','3700079089',
		'5150024136','7045900558','1111161015','9396600585','36382405640','88810911010','2420004491','4440013868','69511912033','61124736760','7910052774','2970034145','73052152101','2220093009','2410011719','7332103999','72225296576','74941788066','7192100331','3800016967',
		'3500044678','7020054066','7326000009','6731200512','2400016287','2550030401','5150025537','1600047862','7265500210','3100000731','72225216710','1600049674','7910022418','7680851558','2073513671','4300004600','4670472075','85331100396','3500097229','3760027095',
		'1111161120','2500001067','3663203856','3890074074','3270011058','4470003342','2900001613','67852337548','89162700928','2073511025','3100080923','2970000135','7766112321','3040022130','3800026595','1707713268','5100024543','72225296020','3760005290','5150005711',
		'6414404236','1450001822','3400017416','3600054264','7232012441','7020083812','5000017492','3700098440','2100061166','2073511262','63810267776','1312000800','2740047053','4133313848','1370077743','4179060030','3732312528','2700037890','63221002749','81829001797',
		'5150072002','5100027655','7910078566','82927421920','2100062566','1122564384','1111100635','4460032337','5150025565','3600019564','7192191157','88949748675','5200004448','7457008005','3000066507','3800034999','3485694068','5100021890','1300079910','88885359022',
		'2100061969','61124736759','7027275000','4157003905','72225216057','1920002845','3400013453','4200016222','5100014800','4430012082','5543762917','3700097078','4200023710','69899780905','7100714159','2700000105','5000042884','1116211201','1780019500','7910027807',
		'2900001615','3700079619','1312000041','3620001364','1600041729','1450001810','3700098208','2073516128','5200004701','1081829001104','7066242404','3800019988','7756772110','5210000698','7418245991','7766112511','3400021466','4300005801','7045900910','81829001105',
		'8259263129','3810013039','7100772942','7271403650','4600028875','7680800684','1007343500201','2100000028','3600045127','7027223204','1960004400','7592540124','7218063811','1004082234537','5783602069','7910052836','4610000164','2100005191','7119000675','1600014645',
		'7835471728','7127956542','2100004876','89386900517','19600525195','4000051296','5410000036','7045900555','3600051511','3270015524','64243412205','22200951040','5000043670','3210003404','7756725420','1007343500230','72225296504','3600054274','4154824277','5000016980',
		'4000049753','3800013859','2620014484','2073516112','930009540','3270011055','2100000729','2700038049','5215905000','4144910821','7590000326','31031032501','3400056002','2700038042','3940001971','7750700503','3800019986','3760018717','3890003031','5200012325',
		'2260002027','2100006067','7910090207','4132100580','1070015671','1450003133','7101204105','3100067102','3620043097','89162700927','89162700250','2310001405','3100067001','4156500024','5020055410','2310014354','3810018977','75703751524','2420009457','4300007874',
		'72225216068','1500093572','7265500211','3800011897','9023150642002','1004132210871','1530001466','4154884204','7020001065','4100058794','3270015526','3663207894','2700050006','70559901164','7910076157','3100010009','3890003014','3000027113','5100019575','85331100398',
		'3320015020','7020083811','1800000183','3600041313','3040079415','1111107626','64034403164','3600051355','1370047520','2550020532','2515505705','7633842813','9955508520','1600016349','1800000236','3500051092','5410001280','4144930254','4850002106','1600027707',
		'7756725424','7940050740','22200954416','1084024313104','7066208272','4470009680','2550077456','1450001990','7403004565','2410070495','7218056667','2900000822','4154881332','2073511011','4000051137','7046206249','8434867726','3940001602','75090311287','4138791785',
		'2200025588','7457030517','4832103530','3800016252','3800022242','1862773917','3980010811','2100007714','1003210005528','7630173011','2100073346','7040900409','3100000720','3077201467','7064001086','3732300201','4240020058','7119043680','4100057650','1114110687',
		'5000062232','1600016685','2100061144','1009642341310','4430012391','22200954409','1113217089','61124736766','4300000791','3400021577','7092047658','85961000004','2900001903','1330060630','2113100032','6731200513','81829001794','7756725004','4112907700','65762241273',
		'9023150640102','4150000052','3500052123','1900008344','3700060554','61124736764','2100072900','2983910024','2310010843','72277620041','2580002040','3663203908','7756713228','7020001067','5410011722','5100021911','2880028078','63221002480','4129440272','7101207503',
		'3620000300','4980004325','1122564385','3100010976','4141942006','2100065655','3500014171','2073516172','4300028543','74759961826','3600053741','3620091411','5100007546','5100024838','4240001593','69899781032','4300004981','88810911525','4171682062','89162700931',
		'5000087067','7778202729','2780006356','6731200215','2880029051','7418247339','4800121337','7261316044','7240006047','5100015252','4300000085','4132100660','8434871012','19600570834','4150001149','5150002512','61124736761','7119076185','3760023013','1340945132',
		'3338322101','2113190557','4116700841','69899781038','4430012083','3600041314','4330161169','3600054271','6414486802','3890002058','78774823215','7680800292','63641200928','3545777635','3000056849','1450002295','4300000842','7756708516','7040900402','3500099662',
		'5170020615','1450001938','2200026361','2100068591','88810911061','3700074919','1740010039','2100065493','1450003129','7218056740','1114110677','980000760','81829001260','3400037924','3620043056','2240064015','7910077250','2100003749','7590000741','3010033460',
		'7680800288','5069511912010','3320011538','4060010971','5100015547','1500096008','3040022134','3000065800','4150873445','2898910065','72225250120','5000001501','3980006225','2880029311','3010047241','3760048692','4440013810','1300001110','5000050288','74759941429',
		'4149709169','2780006794','3700085927','6233882291','3900056566','980089371','7023400411','3600051353','2073509764','7342060124','3700077326','88885300057','7015788277','2340005359','7684000325','1780010045','3700077809','3760025229','7092047672','5410001110',
		'3377611102','3500099674','3800020035','2310001407','5410000050','3940001590','1450011127','2310011130','7667705907','5480042335','1380020030','1862710138','4460032339','4610002048','1115605911','7403008182','1600049415','69899780917','81829001867','7457000700',
		'2100000926','7343500036','7447111393','980000292','1312001280','81590902085','4300000840','1450011128','7590000742','4112907782','88491200681','3800019993','3620091401','4300070992','7585601120','4200044357','1114110630','1780019335','1004149713355','3600050080',
		'3890002060','1007343500233','7184009202','3270015230','1114110229','7265501123','2800029430','3800023458','1600043980','2400056667','4171623215','4132100620','89386900071','1600014943','3700010778','1114110670','2018900120','3800005239','2900007313','4000046408',
		'81829001806','3000031407','7142106614','5100028108','5200004233','7100706140','1750004410','1780012088','2073516170','7495618055','3800019935','5480042351','2310011781','1800000186','5100017965','1780010975','3700097588','1920087870','31031099880','4470001165',
		'3700074958','1500093518','5100017960','4144940178','2100002696','3700086212','4132100565','7457008406','3400035045','81173707938','1800000318','3000031595','980000787','3700097077','4100000271','3620021923','9990089196','3150673901','4150001042','1300000264',
		'2310000099','5210000508','64286311057','1800012447','3663207837','7940050720','7457065058','3340005554','2898910079','2840049557','7766100316','3000056083','1530001459','3000056860','1114110334','1530001476','60265229950','4116705224','64420941270','2100007717',
		'4149709730','1084024313103','2073511013','2073511020','3700082876','4330161165','4300000843','3320000322','81829001902','7007400336','4175500211','2970000139','3700082843','7218063245','7020001063','2780006467','3700075725','5200004427','5150024395','2150004217',
		'61124736758','2310014358','5100023317','85961000015','2100061527','6233891101','1900000305','1450002526','93000077','7192121578','89162700925','5000050423','7040900182','2100007328','69899780907','3270011364','3600047804','7146428040','3800018201','64034402106',
		'1600014733','3150663512','1258778563','3700082934','5020050001','7218063244','7265500131','3800024679','2310010280','2983900092','4132100648','7255462027','4149707377','2000010591','7457061458','7184009204','5100023177','88491235608','5200004755','3400093934',
		'4980013836','7040400103','5480042330','76211162291','87037500501','3700074907','3700077136','1299340109','64034401055','3600019304','3700057523','4300004992','67852303850','1600040989','5100016779','4157014631','1004149713489','5100017968','1800000757','81452101197',
		'88885359063','6414428304','64034401117','1901461097','1600015972','4119691013','5250005005','1480031816','2900002329','2200022578','64034402060','3500014477','1780017423','2500001068','7040900405','7008504692','2800024497','4460030900','2018900121','7027249139',
		'1740010668','1300000342','4149709547','8768400512','2900002333','7585600051','7271400662','3010010034','3810100187','2100005172','5150024769','7232011413','4149709244','2100065886','4144910790','69899781022','2004175701106','2400013207','7457052400','4175500850',
		'7590000736','3700071486','7667705716','2073516132','1450001418','78978501962','1340991810','3700096262','3485618601','4149709744','7590029750','4133533292','85225100009','1600019360','4144900210','7101201050','4000052560','1081829001149','3100000424','1780016543',
		'5440000040','2550030431','3600048596','3600049062','1340945133','4610000113','7261373982','5100022065','3700087073','2100002689','4300001050','7218056638','84223400098','4980003957','9955508051','5150006824','2310033582','4132100722','7261373977','7232013388',
		'3010012101','1900008501','3600053306','1450001595','4200087467','7007453432','7910083631','7240000728','78774823012','7940051910','7630175249','2500010135','7910099376','2580002360','2100012283','2100061526','2460001003','930000070','7142909849','5100021241',
		'64034402194','7585600120','2400025265','3000001240','3600047747','4142001623','4850002107','7066219664','5000010043','1111161117','64034406006','3700097585','2550030418','2260094752','3100020003','2260002026','69899780954','4800170592','5200004490','3600046961',
		'2400052405','4149702338','2970002142','3890000407','4149707371','64034403758','4460030954','72225238600','6403440311','1340912844','4144940326','6233877961','7835470045','2460001700','1370001726','4116400845','75061016106','74759961857','4300004353','7127956512',
		'4116400821','2310002452','1800000512','7046206161','2113100027','9955508589','4610000715','3700094726','1114110260','64034401150','64034402096','2840001541','2310070295','8768400414','4300097825','1340951537','7299948323','5200004458','3320011534','9003150658705',
		'4132211087','980057102','1600016366','72225250090','2073516139','2880013069','7910076161','2310010978','18473900149','2550020225','5100021913','7218063718','3500099661','1004149713340','3800023271','1450001977','3800019951','2310000157','3620000468','5480042334',
		'4116717101','5100017963','1780017957','3100012021','2310011926','2700050008','5000077610','4133500098','4300001662','2515505704','2500004764','1254667614','72277600149','3000056856','1360078911','69899781024','4300004170','1780017744','5150024355','1111168541',
		'31031043015','1004480050172','3760046221','5100021914','4610035339','3700075261','1800000731','4149709789','3600046651','7020050422','2500010141','4175500703','4142004854','1800000058','7940008633','63221003204','1600049455','3700093139','2100005440','930000171',
		'5410001830','7265500103','2410078940','64034401278','3600010358','2800046123','5210000696','3000031133','7261373915','70559901510','2420006119','1780014310','7940026654','4200044175','2100005316','2260093752','61278110118','3905972884','76211188813','2260009187',
		'81032601362','4175500219','4300005788','7192121214','81590902086','1740011845','1700004620','3700094725','4175500860','69511912098','74747900007','1081829001932','1312000278','1004480050311','5210045687','3620043068','3600043616','5000062241','5150000026','7218073366',
		'5210000442','2880028972','4112907705','930000653','1081829001919','1450003158','3760080794','4470008709','60265219913','2100002483','1075090310964','3120022670','3150653302','3600050407','2800021580','2550030442','5100017959','4610002046','4175500212','75031004202',
		'63221002240','1450002346','69899780932','1800010681','1312000833','75090307759','2880029296','2310001538','4171682058','5100012474','4000046410','4010000473','4300006002','2073509637','1901471257','88491235948','4156514024','3010010069','1530001450','3760045088',
		'4610000221','4142047149','7047018511','69511912038','5150001229','3800035802','3700079008','2200028098','3600047930','64034401677','3120000191','1450002253','7027223208','2900001665','64034402104','7007453806','3600053589','4132100724','65827620256','2550020540',
		'3800011895','19637000245','72225238598','3760036040','3800011480','7045900923','4132231141','6343589164','4200043211','3800024712','1450002388','7046200632','5200013513','2100002484','5100015548','1111168445','3600048536','7265540103','5480042332','2073516138',
		'4950812620','4670473910','1450002402','3800024902','1600027706','4460030037','4149704010','41625325403','3980010959','4330161167','7835470185','4133392948','2260001986','4149709462','930000074','3980003858','18473900226','5410000582','78978501963','4116400022',
		'1450002283','930009681','5100024920','82927421690','2620014455','64034401028','4450033902','2310033581','2310013265','2100005176','85225100010','2550010851','3663207984','7756725005','89162700300','2400024114','1007343500004','1480000765','2400025056','7173000741',
		'7585601113','4156519388','2580002242','4179000225','2529300439','3700082836','3400056043','2580002013','5000041518','31031031854','3040079268','1920096317','88133400970','2898910103','1700012003','7255411018','3000006507','2052518367','9451442914','8819434061',
		'73475600002','3800034389','1370084807','2200022584','75031004217','1450002270','7101205050','2780006754','2880013072','3400049063','2340006866','6414404828','3700075415','4240002554','7265503219','75090307754','3800084563','4300005794','3980012111','3940001910',
		'7680800878','4670472065','5000040937','2840058756','7110000021','60265218659','5100022211','9518801591','2200025583','81829001908','3700076820','1450001993','3700076560','1600049432','7684010015','4100000332','63221001296','2200023377','2983900477','7064030007',
		'3760000641','2550020536','3700061789','7040900401','4450033903','3400043228','2880029261','87037500500','7218063717','2880029161','4133500053','3400023900','7101207510','3500097008','1007343506080','3000045052','2150004202','4133500176','1450002427','1450002377',
		'60265225731','1530001421','4000051306','2900002386','7092047909','3800029171','602652185335','7192100332','7585600170','1708287633','7218056624','83418310024','4300008217','2007343500304','2970002145','5000050424','2885205320','3600053362','3400072389','4116400221',
		'2880029395','2880030160','7778202721','2310023526','7066242405','2885205240','1004082201174','72225256109','1600041727','3600054869','1450001497','4112949156','5210000218','7940020570','1708287979','2073516125','3700012769','4132100605','72225266577','7766104813',
		'3100010130','2310014355','3338390203','4180026100','8434830021','4157050003','2220093008','3077200900','4330130519','4610020303','7910014855','9990068375','3000057051','7910000313','7020083819','1800085030','1116210208','88949700110','2880016305','4369533556',
		'4300004347','5100012767','4119691121','5100021221','3000006153','3800020065','7910075994','2550020421','2400055045','4116700133','9990086780','4111800902','2740000024','81213002055','1800000724','4100054111','3620021925','7910051023','2420009433','84024312260',
		'4133500054','7064001347','89762900014','4000052603','1780010978','4133303761','3400029655','63221000505','3760014879','4589308053','1800000079','4119640482','7064002183','4000052558','4150883446','9990091216','1600041550','4610000125','1450002435','4132210921',
		'4116563033','5100021958','1600049465','4180022600','30234089456','76687800051','2880029031','1075090311171','2550020493','4300020433','7265503220','3000026291','5200005048','1450002275','77098130228','7585613901','2880029390','65827620278','7940006672','3900000800',
		'2885205160','3800016254','7940055020','1070008813','75031004213','2310012292','7766111792','1340912841','5150025518','64286311054','3810015492','7192139569','3377601165','4149709470','1780012598','2983910017','1450003131','83418300714','1780017676','8133400299',
		'81829001909','3000031596','3400031273','1340900004','5100015257','1258770320','3760025228','5150024163','4200087445','3905972984','6233879553','7040900403','2220094152','81162002127','7271400248','4440013830','2260093252','1370096720','1600040212','2004175701421',
		'1920081145','67984410456','8434867735','3120001605','3800036702','3800022182','36382400840','7287881037','7097093282','2400025199','2880029301','5100027975','4650072807','4171691008','3700082835','88133400048','4300004929','3760019122','4111800970','3000065620',
		'7585600130','86174500001','5000053062','2100003751','9003150640205','4300000075','7020086445','7680800847','7192101757','63221001220','4000004432','76211162285','9518800233','2310011785','7040400435','2260022887','2100005480','4116741251','2100007844','3890000405',
		'3600051397','1116210250','1600046702','88885359062','3000056205','7007467012','3040079392','84024314407','2410011915','3338367101','4610000107','1708287706','7127911169','7294075600','2880013140','7020054012','3000057341','3800056514','2780010087','1500004832',
		'31254662182','2885205220','3940001603','1090008015','7357053049','7756727105','1081829001920','60265217256','7218063716','2260002950','1600049675','2400025196','6233899056','2100064199','7680800848','63221002885','3980013688','3800076605','3500052152','3120022097',
		'4460033910','1380003053','1009456202561','3076819607','69511912089','3270015900','7990000157','1007354182022','4300005390','4157005699','75166610705','2550020527','3500052155','7261346303','7332100024','3760048058','3800070274','2580002411','2780006373','4144947306',
		'7023406415','7274562620','64286311056','4133500065','5000023902','85000376668','6233802034','4300005401','2983900476','7142909523','7680851709','3077204176','7020083777','3400021467','7045900529','7910052002','2740000023','2570000388','1481322475','8660022203',
		'72885202800','1004149713372','7064001087','7457048259','4180027100','75031004200','7020001066','4610000771','6414404718','4950812612','7299944323','5100021330','4300005001','7101207505','3700096252','7064063023','4330161111','1740022321','4175500810','1450002448',
		'4150001150','3087744800028','5000052523','1600049461','4610000165','5250005009','2073513639','1481305281','3150663517','65827620267','4369500354','2515500019','2620046370','4670409846','1700006807','4175701875','5150000163','4144940336','7829657913','2400025237',
		'7101207511','65724300039','7326000008','30234030271','4670406850','4157005537','3800024237','1363602065','5200012251','4150880307','3000065980','7940050730','2400001623','1481305299','1084024313057','2073542091','7265501122','7790050242','7119000146','1481305286',
		'3600051668','3760023115','3600051518','3400070103','4200044387','2880029661','7829618209','4175500830','2000011196','1380019002','4133500125','1600030630','2100004040','7490832620','7192134486','3000056211','81829001866','3338360503','3120028127','1600035794',
		'7294075801','2700060716','930018400','9007274508985','3800052447','5100015076','7829621169','3800013865','1600012690','4133500034','5100025302','9955509805','3500097231','1600049413','1007774527480','4164690024','6731200527','1780018749','7726009625','4149709756',
		'1254601136','3700034087','4150077404','4149707902','3400002010','9451442913','31284353637','2550030440','6414404335','1450001991','4530069706','4610020318','3000057041','2310023537','5100024830','3620043159','7240000720','7354130561','1901461089','3700078418',
		'1004132210861','4300020552','7119000786','9782966011815','7127956266','2073542099','7892933587','4610035403','5000045557','1001397103000','7232011823','7990000169','930000682','3100067103','1004149713439','4111800985','84024312248','1450003168','75379200201','5200013532',
		'4600029548','3663202808','2073511294','4132100675','2880027850','85402100831','3000056166','7101208104','4132100661','4157005442','7592540120','1004149713441','7431235710','980057221','4300005878','1007343506050','7064001550','81829001865','7101207523','3760023436',
		'1370085531','3810017399','3600054505','3890003039','5100028520','4144947179','61124736762','7265500180','1090000504','4144947206','1003114201007','5150025151','7007457539','3010049051','2580000062','1340951724','7592540138','7064030027','2073509643','1254601152',
		'2100004898','2400025055','81213002053','1481305283','3000045130','5434722001','60265218660','2100000719','5100028417','1115605927','7064001551','87037500502','3700040365','3507449936','9782966163250','1370052278','4149709755','1600026028','9518800505','4116400048',
		'7092047979','9990035225','60265224966','4440015650','1780010973','1600016682','7910088257','3320000206','7332174522','1114110674','4200044280','4164612354','85000376660','1450001493','3600051450','2500010151','7829650298','3400011032','7066219503','7626500900',
		'3800024659','5200012206','5480042331','76687800052','9518801590','3760018376','2260001991','4650000315','65827620302','5100018723','7829616919','4116709444','4440015770','3800017592','2073516129','2420099942','4610035338','1081829001936','1600048927','5150024321',
		'7339008435','7756722700','1070050181','1600015764','85290900342','71752481102','1600014642','7829636159','3120002931','3980008513','1258778913','1111168631','2200029060','85290900330','1450001351','2580002028','6414404717','4132210949','9518801029','2200027613',
		'4133500086','7097049133','2580002296','1111515658','3077201608','3810014181','4149709282','7343500037','1600041314','3600051702','2100007843','72277620002','7630618080','7007468053','7007466969','7940019720','83418300099','1500001375','7910052030','3890000940',
		'1450001948','63782210400','4300005007','7008504946','1500007360','4300005402','4180020089','4610000222','7457090942','63810267780','70559901222','4133500064','1111101216','7040400149','76211162281','3663207896','65827620258','4111800908','3320000136','3400056004',
		'1075090310967','3600053748','5200004915','5849672302','3700097596','5210054736','65827620300','2240064016','3400017121','3000057064','4610035341','74816262110','72225250180','1481305289','7046200822','2983900093','8819434060','7119042183','5100023262','5480042342',
		'7495600000','4240041578','3700079468','4132100625','1920095872','4116709381','2580002320','7680801067','1740022252','1113217111','7704311355','4127102763','1900008503','5000053032','3077201301','1700016917','1862770311','7046206211','1500007000','4132239021',
		'63221003731','4132100574','2310010842','3010033289','1900008342','7940067027','73475600008','1600035738','89162700929','4300006953','7457093197','7592530697','7020054010','2880013068','7940026093','5100023320','3600047414','9782966061301','7101208006','1004149713378',
		'3940001688','1090000330','2580002025','5150024358','7066212610','3800029175','4100073117','2400001996','81097900737','69511912097','1787370513','4132100575','3600053353','3800022174','1001397102102','3010078466','5100006001','4369500357','1500091212','5543763112',
		'7097049131','74747900105','1780016975','7940026570','980012324','5210000446','19600524395','1530001422','7940050750','2570071139','7684040021','76211126168','84223400139','3485678886','61300872592','5200013514','5000043710','7047019074','3980010283','7756725434',
		'60265227964','3700078784','2700038827','4832105401','7261373916','5000029431','2700038811','1001123349205','64420979349','3980010288','7023012722','7027248011','4440012568','4175500808','4300005876','1530001420','7910052778','83418300712','1480051647','4832104991',
		'1450001967','2113190538','3800084578','1708288524','61124738293','4200015719','5100020655','1740014007','7240000660','2052511841','2260060005','3500097162','2550000368','4369500351','3940001916','3700098110','1600040981','4300000550','3810100189','3000016901',
		'65827620220','4133533218','1075090311196','3732300202','41322353112','2400002771','6414404702','72225236619','1111100637','2100004268','4145811704','4300000555','1450001498','2880029410','4180050153','3000004090','7910091583','7101200004','2073516171','5100024508',
		'2983900475','2550020358','2420004789','4450033916','1901461094','2260022329','65827620292','3000001210','1004149713343','4531036480','2550020529','1500007655','4149713635','2240064009','1600018696','1740010268','3700099828','4300008220','74747940002','2880010312',
		'2260022339','7015788269','81097900825','5150000025','7299942023','930000048','5150000055','3700090797','1450001985','1254601137','7192101758','7142100686','74759941866','1600017999','1111100638','3620021927','4300000076','4223831220','3980013077','1800081772',
		'81829001470','2073513632','41736010137','1600050940','7064001088','980055401','3700097590','76211128788','3500097335','7097049360','3940001689','7255412262','5210066856','3400001878','5150001701','4149709736','5100024624','30521525100','1450000501','4144940320',
		'3077201518','5543762916','70559901216','2550020534','1500004799','3800026249','2400034286','4600029549','2066200618','89386900072','2073516130','1600016915','5000050426','4156500017','4300000549','1600040986','4300000652','5100021908','3663207895','4142012705',
		'1708288317','7829614409','2570000140','7046206162','4149707610','76211100109','5170075708','7130000080','2700085250','3980010285','1920087871','77034018667','3400044611','1740014038','7910077482','85407400621','85402100821','4430012111','2150097602','7774529413',
		'1300000346','89162700251','1111168912','2898910089','7040900407','3760027200','2570070953','7990000121','5150024322','5100028207','3700082879','7127920172','6233899059','1370069737','1780016898','76211125933','7756719342','4200044253','5000057501','3087744800868',
		'4300004198','5100013459','3700001798','7092047806','76211128784','1600016490','30521004080','3077201135','3120023473','2880029091','3000031597','4300005386','4460001234','3500051089','3500076441','4142001392','9955508651','1600026470','60502100222','2410070562',
		'4300004358','1450002161','5100028519','6798108794','4119691032','2780006789','2066200607','4600013006','2400025200','4145810556','2400025071','60265225283','4179060029','74759962459','5410000970','5400039327','5210003780','3980009021','3000031598','7034600009',
		'7101207504','2100002317','4300000047','7142901446','1450002700','81452101094','7630172227','3087744800018','1450002432','60265218064','7097049356','1340935150','61124738135','3760023287','7007466912','2400004724','5100000067','1004149713388','1500096019','2310011927',
		'3800076603','7064001567','4144947032','1001397102101','1004149713387','3600051334','7354130527','4133500061','85290900345','3000002913','3800019877','72225266507','4610000717','63810267775','1360074902','4300005779','7703401149','3663203966','3100067005','3270012019',
		'4850000629','3000057240','7115907311','3400022600','7774529190','2550020212','4149702531','4138753022','4428497630','2260092742','1330060631','2073513646','7778202728','3340040148','3940001593','81213002054','1258778899','4116400041','7681134352','7756700258',
		'7680800047','7064001951','7097093024','5000059774','3700074908','3338324026','68142102101','4116705723','2073516174','4116526291','1800045675','2580002622','2529300438','2550030443','63782210900','9990071526','8730056010','3000031182','7007457240','3663202016',
		'7756725438','3000032189','3600054226','7101207508','3700077133','2800050604','9003150663503','75703700045','7892933588','3940001901','4132100711','5100000366','2113150114','2100061659','4430012180','3800026179','4000057640','1258779298','4149709739','8133400301',
		'7064001611','4149702532','3000056947','88949704437','1258778833','7101201330','4300020040','2880013065','6414415075','81097900125','8730056011','7940055300','2880013067','1340951627','4173608001','7703401140','4175500709','5440060050','60265224991','3100067100',
		'3890002904','2898910107','3800019862','2400002125','4133500068','2073516152','5100007688','4149709777','4000046411','7940063428','72225260241','4300004181','7590000533','1111140784','88885300115','1600017113','3760051688','2880029281','7261373917','7007466639',
		'2880013136','3000065320','4300000895','2073516176','4116705810','1600018003','2840049568','5100027103','1780018457','4450098912','1800000014','5100013865','85656400801','60265224969','2310010840','3890004170','4175500863','3800084577','7110000477','4144947149',
		'1708288811','7240000659','2780005374','4610000122','7265500134','4195317510','2983900013','1450009334','7684010132','4149707573','3810100044','4175701861','7680800625','2200022015','7680853355','1500007365','2550030427','76172005110','3600050091','4610000163',
		'63221000947','3400070077','4610000226','7667705905','7258672135','4146020007','7940026094','4116400841','2310013524','4300005667','73475600009','4300006371','1780019035','2983900095','4118810184','67852303844','7680800882','67852307031','6414404707','3077205661',
		'7027223202','64420900162','5000096375','82927421880','4110056800','2100007659','4149707397','2100065478','88885359067','1111101414','4119691401','4150880087','88810925227','4171687550','1450001485','5833618002','4410019300','4610000106','5440000024','85402100819',
		'4149707877','3890001473','7240071124','3360042061','2740000022','930000030','1708287631','4132210931','7940035297','3800084560','2880013064','3600051703','70559901323','7756725440','76211179575','73475600010','2340001071','1920099833','4150880678','7321406618',
		'5100015236','1081097900429','81363602064','7020051323','2400025197','7832280000','3100067117','3320018731','1500007659','60265217851','4149709279','2000012138','4145810534','4000051302','5020055730','7681104209','7056098467','60265217285','1450001574','2550020044',
		'3100067007','3338366001','5000057800','83418300705','87145900023','4300005385','4610000968','7940025190','4175500822','4175702148','76211116543','4280010953','5100007625','3620000550','5150010869','2880029485','4300005091','2900002077','7064001566','71941072610',
		'2200028023','5000050433','4200044359','7348414420','5000057934','1780018015','3120000569','2100072902','1700001841','4149713636','65827620262','2073511203','4119691068','4157050004','3810017839','5100023282','7119010804','84223400234','7265500183','5480042340',
		'4200087572','4149707572','3400070102','3210002631','3600051335','7680800022','1708206003','2586659190','3800023144','1075090310965','7339067204','19056912473','2400025072','61278110104','3100030396','7585602409','3700047892','1114110343','3600046995','60265225944',
		'4127102768','3940001914','81213002086','7119000674','3150663901','3320000205','1600049464','4144910866','3010018201','7940026280','5100014002','76211126166','4000032252','3338366003','18473900219','5833618004','4610000714','4133500168','7040400278','4300005000',
		'1500004680','2260090133','2400002136','1740022322','1500004500','2410011716','2260008161','3000031181','5000050203','2740046516','2100005488','2983900015','1380010393','2240062374','2515500046','4000001244','3810017396','1708287635','7110021157','5210002872',
		'5100014878','2100070147','1600012928','4100000490','9003150673923','2898910109','3700079743','4300005796','3000031684','4154862824','1600050330','3760003480','7020051324','1115605094','3760067437','4149713745','2150000300','3600051826','2073516148','7056098784',
		'7045034381','1450002513','3700092032','7064001786','81861702165','930009011','4149709740','83418300699','63221002729','4116706626','75090311292','2400056515','86048500018','4670406332','1500007050','70559901321','7261345120','4132237505','1450001487','2400051068',
		'4133500127','3700092856','3600049394','1600035795','61124738131','76211188818','7431253545','3120023027','2898910418','7192101756','5100014880','4300000891','5215970475','79452200173','2550030410','2073516126','3700091093','3800019781','74759962208','2800047731',
		'2073516140','4111800968','3980003280','60265227959','18473900264','3940001881','19600524241','7626500720','4132210922','4300005384','2580001941','7321406617','1004149713342','4112963240','5210009020','7457014860','3810017644','1111165762','7045004191','4144947397',
		'2880014503','2310010841','64243412212','1450003167','1780017999','1500007051','70559901512','40000116037','4132110534','89000015002','2000017681','5020000172','7192150313','3760010471','60265219909','1600014711','84024312329','19056912310','7142909943','2400057819',
		'2310010269','7680853356','2073511202','85407400605','3400099662','1740014052','3620043058','1600049435','1500004828','2580002039','3700060298','2410057466','4300020471','3010010403','4300001659','7348412309','1450002527','2073516123','7447100610','7047565714',
		'7778202725','1111168450','9741901001','3800026065','7457093586','1312001036','3760006614','2073511284','74747900060','4195317509','3500097159','3600051412','4132100721','2310003069','81829001833','7032801059','2580001965','3500068045','81097900824','3700076180',
		'5200004265','2310002451','7766100313','7910053387','5480042348','75928333445','3760000687','1600017996','7192125314','7045084191','5100021533','3077204170','2100007847','47726004404','1901470072','7047565713','7940006671','3800026242','4300000552','3700095683',
		'3760023884','3338311907','7046200634','4610040066','1901461206','77034081951','2410011097','3700075203','19056912312','3076833043','4600013178','4610000121','69511940232','7778202720','2200022031','3400056170','5100021518','7066223023','4589303305','1450001780',
		'3000056806','82415060148','3980091156','1116210235','7101207500','7648900478','4369507511','4330130577','4149709758','77098103408','4670462070','4116502220','1600017934','2898910101','3600049680','930009109','89520300109','4149713555','72225230330','2898910091',
		'3700078997','4149707662','7878350610','3077201569','6414486804','5000017121','4138753010','5100019578','3700022419','7020051327','85420800501','5150010860','2800080320','87744800367','89520300101','3500046383','3810100191','61300872815','2066200613','3100000379',
		'4156514116','3500097336','3010022869','1500007001','4116400045','3000031683','7119076181','2970002125','163782212131','3700074909','7910076160','3338380104','7663752968','230002','4650002274','88885359037','3700074959','2550020165','3760026042','5100021244',
		'5833618000','1708287740','4300004149','3400035112','1001123349224','4149707443','89000012600','1500096005','7663752972','7940034218','5200002027','87744800717','70559901344','7663752985','7942622107','5100007623','5480042343','5783616826','31284353640','4133534955',
		'81529400021','7064030012','7192128605','1081829001148','7040400013','4138733127','4149723751','7431255366','7704311356','4440017300','4300000892','4300028589','84223400232','1530001496','1600041625','4610041103','2100061928','7232011297','1004480075100','4369507512',
		'4300020556','87145900107','88491224926','7663752973','3320000260','1500096006','61124738290','3700076562','7040900431','1862770345','1081829001933','74747900118','7064001238','81829001672','5215970477','1600017944','2073516155','1480000648','7684010098','1300000814',
		'6414404709','7040900223','4149723728','3338390424','2240064010','88133401268','1114110342','4133500063','65922318120','69899781039','4133533458','7255400191','1530001467','7192148448','7478091135','3980009198','2100065492','81097900785','7116976000','1600030570',
		'3100067089','2570070946','76172005010','76211126165','4850020497','4116400066','4200015136','1370001005','1450001489','1159650210','4119691505','1450001972','89520300117','7239293016','88133401269','89520300116','4149709341','4610000940','4180020090','4300020472',
		'89520300129','4300020439','4179060013','4175500707','1750008120','75928367321','7040400453','75764561460','4149723785','895203001288','7585607870','7940001743','3100000912','4116400026','3000057021','7007458061','84223400163','4149706576','163782212137','4000052541',
		'5410093602','7910002840','89520300126','3700090833','4149723729','7040900474','7046200635','7294075701','30025810931','4149723753','7940006125','5170095315','30521004082','7940026560','3700074987','4410060990','1007274554277','1600049457','2265570196','1600041751',
		'81032601422','3760000686','1600016697','5849670123','4190008901','4300005797','4146020279','85641600071','3700087087','3700058645','88491200248','4133500171','2500010131','7681114440','2529300172','7471408846','3000056202','74747900115','67852307033','7321406134',
		'4149723730','7891170800','74747900155','74747900150','5003150669443','4149713559','4149707888','1920099100','67852303846','7680800649','4173608002','2580007218','72745668296','3980006224','85921300500','3760048258','3600049693','7218063247','1450001412','5300007170',
		'1450001276','4149723732','7684010058','7348414400','4149723779','1111169114','2460001001','88133400696','5434726010','1111101215','76211130194','3800011286','1258778536','5300006898','1116211231','1090063384','65827620271','3077205663','1450001491','7990000123',
		'2580002398','4150082772','5210000234','3700052365','7648982813','69511912039','3800024377','5100017962','1500004479','1740010973','81957301067','5000056790','19637000244','6798108892','3150662901','1450003159','7119000183','1769690704','1111163852','2260000111',
		'7910083410','4220552333','3800026419','65922324064','9990056156','1380010172','1380012007','7142901230','5100014802','2420003829','4164612415','4300009103','7681134360','1114110534','4154862431','7910085176','2880029486','1370011830','9990010043','4150108321',
		'2780010079','4300008035','4149706607','4149723734','1113216871','2970002180','2240039375','4300001614','71941070610','4150880312','1500004459','2580002011','7940033943','2660089334','3700055193','83609391122','60265228859','1800012594','4300000772','8298801096',
		'7287827570','1380010072','4116400843','3600054495','1901470071','2310011334','7484773716','2260060113','4149707107','3600049695','3810100188','4300000619','4133400154','84223400099','3940001968','7585600054','3338390402','1780010976','3600049987','3000032093',
		'4164648376','84223400097','1800011709','72225260244','81829001832','5100027916','1901471245','4133532955','4149705905','3700096258','4149709920','4300020030','4138761358','1500096021','4138752458','1313000698','7299944423','7766101413','5100028110','18473900220',
		'4149706582','7130000020','1780016967','3760046454','4149707899','3338380125','73891202232','1901471107','4133500177','81829001184','4610000941','4116400849','4300020691','1004480050121','4116709806','4300006064','31254662181','4300020445','3260103115','77098108101',
		'2220025456','2066200053','2400004406','30521004083','2150004206','7056098655','7684000236','3100067002','3700089039','2310011437','61278110108','2550030444','5000004041','87145900320','1450001492','3700000774','1800042697','2570070407','4149723731','4133500050',
		'7261345111','4000058028','5300005365','3600049959','7478031086','3810013014','1340951741','3700027445','1900009400','1007020055469','71631055146','4610000233','3700094542','3600048288','2570000330','1004149713437','88885359038','60265228350','4600047971','980057143',
		'1113216874','1600041975','2420000087','4300020053','4144947029','2410019132','72745663208','72745668654','2880014313','7471408858','1500058945','2850010245','71860497384','5543760258','7218073367','3022300301','1360000079','2700041928','1001397102100','74747900013',
		'4300020442','76211162289','3485694062','3800027346','5410001272','3940001199','74747900001','2550030445','76211188812','1115605093','2970002126','2073509742','3000056093','84024312325','4180050500','4149709469','4300004178','8000050239','3700091092','9300009102',
		'2900007329','2500012057','3700087473','74747906100','1084024313589','5000017195','76211130192','4300006954','4195311450','76211120612','2515500054','7045014891','4280010952','2052597008','1901471105','1450009434','7940053430','3700093043','4116700910','2700038810',
		'4164648389','7990000161','2880013436','5480042344','74941122398','2880029701','1340934115','4150000099','84024314175','7101205555','6233887980','7101207528','3700078449','3360042041','3800019110','1081097900118','5200005063','3890072060','4145855082','7064001565',
		'7458405064','7240001109','1300015160','2898910083','4650001842','4610040092','3320000173','4149705904','5200013519','18473900218','60265227132','1780016989','2073511229','3100000382','2410010623','1700001686','4300020440','602652185540','5200012196','4300028593',
		'2100061168','2983900473','7294075400','4175701463','3760047749','1600015243','3600003906','3022304122','1600048287','1480000686','2073511295','76437591871','76211128790','2100007848','72225256803','1258778361','3600050976','8000051420','4149722419','2004175701101',
		'69511912043','3040079232','4650002667','60265242959','3400099661','3600051535','7040900329','5003150669463','1370071625','1500007031','8730052648','2113150702','1600016430','3800039672','4149709245','3000031184','4100000272','5150000020','3450063251','2880029062',
		'7940034368','7047017541','1600030140','4300000628','3400021492','4300000580','1004980022052','4610000969','1258778362','2310013773','4470010321','5100027802','6414402066','2200023376','4142005217','4610000180','1450002234','7119000693','7119000668','2073511223',
		'3760000681','61278110124','3940001976','2400016713','3663207772','61300872183','2586659189','1780018779','2800003328','9616286060','6798108902','3400093935','1111101040','1300000854','7457065365','82415060548','4138753012','71631055191','7261316148','3700090801',
		'8768400402','7064001560','5100021586','2570000389','5000041580','4162536550','1500099792','3500098607','2800021010','4330161119','4610000781','4119691093','4300005392','980080221','930009090','4300020072','8298800096','3000057042','2550030439','4000057634',
		'1004980022091','4000056091','7756713229','64420900409','81173707939','76211192557','7047014784','9396600998','2073511277','4100057623','4300004168','7929690103','7119043670','3600019566','4138753020','3320014071','4180050154','1004980022017','70559901349','71860497381',
		'1001123349201','5150025527','3400005100','5783602190','5174603372','4300020553','2880013066','1740014075','3500044633','7940000878','7342053064','4138711295','4110080676','1500004564','60265227136','7592530754','7920086705','5434726011','7320289457','2880027901',
		'3010012738','7878350802','4610040012','7590097122','7020054018','3320000093','1780016963','2073513504','4164690023','3150000127','4116705877','3120022315','2800066734','4111800981','3000031685','88885358936','3077201450','7940056319','73891202236','3760013820',
		'1450001495','4430005466','7015788921','71631055185','2970002181','4300005877','2310013649','1480000689','2970002182','7192101759','4164612412','89520300115','2073542096','88491201424','7101210701','5200005157','7192120775','3800025900','5200033876','7064001309',
		'73891202230','3700086236','980000573','4000052553','2100064104','4149713676','75764561310','3700034089','3940011488','4000052617','5000050126','1019101100060','4300020431','4150098921','7064001727','980000733','3760047065','4149709463','4300006952','4300005393',
		'4300020051','1370085656','7940025730','7478035794','7046200825','3760026063','7173000677','7680800846','5000060198','5150025188','3700047415','4171684088','76211193446','2260064614','3077206130','76637548479','4179060031','3120001603','71785410521','4133500052',
		'3338394782','3400014060','4100005014','3810018586','61124739607','72745666193','2073511290','1833701217','3338380110','4138710241','3040079436','9616285454','7929690102','71785415209','5100017547','73891202248','5000011081','3000026190','7878351001','85168100837',
		'7940064949','3810015865','7940058956','4293441003','1500099791','2880014645','2700069034','71860497233','64420900408','7940058987','76211193448','1450002434','3400001873','5000048723','74747900117','3000031280','60265242964','930000552','3000031993','2880010901',
		'5210003824','3022304125','1780015497','82415060448','4149709754','4149707024','88491200247','3700052778','5000057050','4280010921','72225263310','7056098162','3600049678','3150620506','2800061994','5100026923','1500004836','2073516135','88133400966','4118810028',
		'4440015448','7940025740','3400047224','64034403848','72277620003','2780006597','3320000267','4300000763','3700078816','1530001489','2550020043','7240000729','71941078536','7255475477','7294075402','1800042742','1500007658','4133536381','94053','7418245642',
		'1312000455','1708287708','84223400233','2580002910','1004082201134','7192177800','71860497380','65827622223','8730070005','5100027333','7321406129','3133248257','4460032328','76211188810','3700076360','2066200601','2073516127','4300020812','5000057547','36382417565',
		'3760067436','930009063','5150010805','4610041106','4610040029','7101207502','74816262109','73475601072','4138740646','3700079127','7040400434','4610000942','2016922233','7057504049','88810925332','5300005282','31031031805','2240039365','7940045948','4200044351',
		'60265227841','2880029951','5210003822','5210003420','7940058983','1356200004','8133400300','7940046583','7681104232','2000016255','76163520810','4149709483','7299980413','4116735121','7940026655','7940037998','7940031743','3600003904','3600053363','7066219501',
		'2880014312','1116210230','4100000833','4300000656','3320000266','7320251021','4300020140','82927476199','2260060004','980055622','7320289241','18473900107','7192141779','7101210546','60069900156','85000376676','69511912005','2260090150','76211120603','4000052551',
		'3800026968','77098104107','4119691051','7110021147','4300020071','2400011764','6731200574','1600018587','7457091010','4133500129','4119689123','4175702187','7940001343','4000039505','3100080948','1450001486','1450001783','49000023501','4149709281','2580004804',
		'78656055728','1340935100','3760082086','4149709288','3500097294','5150000021','74747900004','1004149713348','82927476241','3800025188','7726009622','7192177261','4369507116','3700080720','1114110215','4470009406','4240041570','3663203464','7261345719','1114110344',
		'81829001868','2400011770','2900001938','1450002347','3700079546','88885300113','7940031744','7703401801','6233885723','1600046701','4146039991','7321406162','3320094212','1500004834','67984410464','5000040696','4145811705','3940001912','2880028079','3600054648',
		'4010000287','4300001669','71941072366','1769690705','7218056502','1450001981','3800024547','2880014649','4142002180','4149719350','5150078228','2800094370','7756700218','1600013978','4116560009','3800026382','3760059206','1708287735','74747910000','2898910063',
		'4589308641','75031006107','1007354136322','84024314072','30234024100','1600030670','6798108798','3760016588','7156766093','7630175252','7299919803','3000031269','4149711178','1600026314','81213002079','5210004985','4179060015','2310010835','2260092674','1450002519',
		'84223400074','7929690248','7007453623','3700050978','954203487','2200023835','5100007624','4149709466','7192166848','88885359064','7092005012','3160402508','1600043064','64420941295','31254663787','3210005817','3087744800750','3120029450','61124739032','7064001561',
		'86545800013','81829001942','7756725427','2200029054','7348419400','7020050424','60265227734','1090008027','71785410501','5543763217','4149707901','3700057471','2898910087','980012122','5200004227','5100025274','72225250200','1600014091','5100013140','4430012085',
		'3400009256','1700002654','7321406105','1530020098','4650021764','4146020280','85000376648','2515500013','4410019302','7023406470','1708206073','1370085760','4531024320','3400005300','1600040996','5210003821','980055201','74747940013','5150000162','63782206316',
		'4132212654','7339061041','5150002513','4610040010','2850010060','3600054649','2066200602','71785430551','4300020437','1500091049','1450001581','7832257700','1111169089','7046200826','1300001332','7020055110','7110021038','7192130298','1284271001','1650054456',
		'1600014945','68907699259','3210005821','4175768042','75611019579','5200004691','2880040006','2113135023','3400005220','4150099279','2400016305','3760037084','2400004725','3600032519','3140007907','7756714169','2016922209','76211193025','4300020652','7592530725',
		'76211162287','2113190564','71860497814','3890001613','4610040059','9003150662405','3074747940001','1370071016','1600045201','2580002020','3940001809','76211126167','3500097872','5480042345','1480064671','2570070755','1500004831','60265225284','4600028873','4530000542',
		'1004480050111','30234085328','3732313502','4280011839','1356261001','81219','65762242386','4175702185','60502160049','4950810004','4223830250','4119689108','4146039990','71860497379','3291313762','7684010047','1004980022089','1600018675','1114110671','7255449821',
		'2113150910','2460001707','4650073809','7209241212','73891202231','4119691000','1600018517','3077206166','1450000026','7144830014','1380010018','88949785697','3000057052','37600301374','980000744','2310001404','3100067006','7116922762','5100021738','4150081843',
		'7218056455','71941079254','4149713677','4145810300','1980000116','4280011838','4670406331','4133500014','7680801127','3120022085','7020054007','36382417165','88885359066','2073516131','8000049524','85641600070','1087668100454','3000056917','75703795062','3100012687',
		'4116400031','7007462885','980089221','1600015087','76211194958','6233891110','7265500111','4300005904','1500007359','3700080676','87037500503','3700079716','4111800980','7057504009','1600017941','3980010289','65922378125','88885300058','1003836140305','89386900518',
		'7990000199','7092047663','3077208823','5250005002','5000073907','61278110119','8000050527','3210003048','4610000224','4300004868','2073511224','4162592412','82040246801','1600039510','4470010296','1500001373','2550020040','1090000024','4000035387','4300006899',
		'1111163039','1600017923','7255461657','4119680548','88885359021','3760048563','5000096364','60265217828','81097900356','3810100049','1750008150','3133248250','3150620306','7940055110','85808900315','76211120611','2850010005','7585600052','8298800006','3000031328',
		'2310014367','4149743476','2586659223','3760048701','1600049037','6414486809','4610040002','74759941042','980080404','4138753030','7630172219','3600054650','1370097251','3800016748','31254662158','4900054412','7940059146','3338353030','5170098897','88885359024',
		'2240039363','4138732791','3320019714','5100015871','5100024542','1800042689','4300000585','4610041107','3600049694','4149713692','2310023529','9003150666302','61278110109','63748000038','2880029055','85402100803','1700014290','5100007501','1380010070','87786900128',
		'5150014127','1708287632','3800018251','3000057049','88399065120','88885359055','2073511292','7248600220','7057504058','74747910006','3500049874','4149709458','3000057370','5210005231','5100027622','2200022016','5100018034','4460032090','3320000092','88885359048',
		'1800042692','76211120613','19056952218','4430005462','72885205354','4300020692','1500004456','73891202529','72885205707','75710701113','5200012937','1600019726','7027223205','3400056001','63221003567','3320010010','3760011289','5434726012','5000057018','7020050431',
		'1070070280','3700079240','1708287630','8298801006','3160401328','3120000413','1700012751','60265227152','7348412187','8000051604','1500096007','2198520058','4610000943','2310001779','81097900775','4133303255','5000041570','2066200600','60265227144','2260064214',
		'63748000022','3600046595','4280046076','3760079179','5150010866','5210074862','3338311010','7066212601','3600001396','1570020968','7104000035','4133500169','4110056656','4000001151','1650058702','6602200033','5210054729','4116400071','71860497214','7891328002',
		'76211138017','1004480075901','3010010595','1500004475','36382407228','930009108','2570070667','88491235941','930000338','2000011313','777459744','7940006296','82927476200','74747940004','3600047915','3890077395','3700078980','72885205367','1708288806','1114079810',
		'3150613402','7079660203','85556911064','3800016774','7832230000','1901480297','85556911024','1500004487','2840004940','70559901481','5200005153','1600016167','1780017584','1600016239','1800042691','3600053594','4300020555','3700080675','7940052806','5100026934',
		'61124739025','4119691012','7218063694','76211162290','88491235618','5100017967','4133500021','4157005331','8000051308','7064001241','3760016016','3890000818','7339067838','4149713633','7040400280','73891202233','4138755103','3760041864','4119691074','1004149713447',
		'3760088621','3600049060','2850010040','2000010165','85650200608','3500046977','4650073868','75703735270','7684048511','7348414450','4149709629','4118810189','71785410555','75031006114','74747910005','4300020074','5100028420','2898910067','1450002280','7232011265',
		'4150880302','5480042364','67852303075','87145900022','2570000652','85650200626','3600051534','4132100713','7265501112','7294075709','6233892944','76211188811','7592538005','3320010011','3140007905','4116709428','2066200096','4610000970','3133248251','70935100013',
		'70559901511','7007457533','3291313764','7940086666','3760010491','1500007369','7940020696','1600015579','30521004339','4133536379','1003836140304','1800042698','2073511057','74747900012','1600018145','5480042341','980020127','1500004534','7101206007','4149705901',
		'7592530132','72225256801','70865600146','5210000448','2880017507','4133534072','3700097070','84024314074','2260001988','1500004477','72225225169','1600014665','36382407214','4149709287','4179000165','3620021913','2898910093','9616211791','19056930451','1600030790',
		'7630184504','85000376677','5150024548','3400024501','4116400077','84024314276','7326000531','64420941085','7047013704','7064001271','1500004813','60265243012','3620043100','2100006865','7684010207','7020051326','60265229952','7119000669','3760049120','3320018732',
		'3100019624','7127920073','84024314411','36382417163','65827620227','7092047668','2885202580','76211138016','2880029399','60265227962','1600050950','4330161191','5150010827','7667705904','2898910081','2066200619','4369507111','19056952219','2260022008','73891202241',
		'7681114122','4300001915','5100023281','3810014495','36382484823','3700048081','1600015973','3980010479','89386900056','87489600068','2880028075','7047018709','1500099901','2240039361','3400037105','4149744123','7680800881','7127956288','5210082791','7007456973',
		'70935100901','7592530138','4149723780','3010003319','70865600144','88462310033','1007020055472','7114066000','7320289451','1114110346','5210015643','7940033944','2400003409','3663202986','4133536380','2620000351','87489600037','4114309080','7756728192','5200033877',
		'7209201218','3760014880','5003150625008','1450002401','7990000303','71785410557','4133533291','5150025578','7339062344','1085961000588','3400024565','7119077604','7097049135','1500000536','7218056508','1111163853','4410019313','4300003227','72225256853','7590000650',
		'5210003021','7891102526','3010019546','7630184512','2800092812','60265227960','1380012006','3810017648','4149707884','4589310554','1500004497','3077206135','1780015499','2400052541','1600041628','7940026700','76211126164','7299980513','7261346309','7680801172',
		'9616227076','88491238656','3600054879','3980013125','4157005466','1800044426','9616286274','7320289243','3700048688','4650076946','7457065148','5300006842','7110021399','2850010025','4119691061','87786900798','1780018702','3700052698','7114055161','87218100014',
		'1085961000548','3500068783','3340045053','2240000190','2620000352','5200012178','5200010239','4138710240','4114312460','2420004179','2004175700109','82927476198','1600029860','8000051304','70559901155','4610040096','4010000474','4149719351','1004149713345','1481322480',
		'4149705902','7008504690','18473900224','4280046086','3800022166','2260002914','61124739529','60265225730','7680800793','4149713682','88491201425','4300020054','1450001494','70559901353','64034401341','1340951767','7490420210','4300008512','4157011070','85961000748',
		'4149705906','7209201212','70559901629','72277623001','84024314770','1800012449','1750038206','2570071381','3500076667','3000031194','4156522741','70935100016','3377609145','4300002165','85068711051','7110000613','3140007915','3600051588','4280046084','3980007680',
		'4300000042','7079660204','7484720320','1004149713375','67667000106','4300020038','67667000107','4127102256','1600041215','31284355540','4149707342','3600054653','3150611606','3700085924','4000058011','2550020106','1380012009','4300000764','4300020032','4800100069',
		'1862711180','5100006008','1450001980','7832257600','8298801106','85000376665','7592530755','3281207004','6233883550','1952155005','81093403020','2073511301','71785415210','7786500152','4133536378','1004149713368','4133535922','76211128786','4116717201','2100007845',
		'67852303081','1007929690513','4610040094','7156766094','2410000139','76211120604','7116923322','3800024909','88399066120','3000031176','4119691415','4300005798','7040400492','4133533517','3400022901','4114312956','8259263380','1070080726','88133400276','4195317512',
		'3360042011','2310010123','7453410164','6731200210','4133536382','4116705620','2113109025','7299949323','5000096929','31254663788','1740014003','5010021016','7756760332','70559901541','7590033333','5783600043','4116760352','5210004225','4127102273','3600001404',
		'78978513837','64786599550','2400052389','4116400046','7590000651','4300028533','4116710401','4149705900','70935100902','4114312010','6798109677','74747940006','60265227742','1500004969','5010021013','3320000416','1001123349203','7940045756','4119691507','3400001314',
		'3000057272','36382417563','4114312576','5300006886','2850010020','60265227961','7940000879','7457013335','2400032230','1600040982','7680801173','68142102081','5000017193','7940007216','3620043099','7339080014','7490836054','81957301554','3500076666','5210049932',
		'4150880317','5000021220','980055624','4589308662','1090008028','7064001557','2900065150','1780018707','3000032063','3338390408','4149713350','5210004642','7320289455','4150873086','4300020449','7484777716','4280010918','5100026916','7940059303','74759961703',
		'4149707018','1380010067','3500096911','2819000739','1007236839381','3545777646','1111546485','2016922235','74239273600','3600051583','2570018036','5210000346','4000001105','3500049110','7940020695','7114055230','4149709424','7064001955','5827621321','87489600069',
		'2880010314','4195300271','5000010209','3120000168','7684010035','4600086011','3760041017','84223400052','5150000065','3340060111','1116210215','6233878046','4300006902','7756760322','7116976010','74747900114','3800019887','3400056023','2400003470','87489600072',
		'81097900358','3320019020','3700075202','4450098467','7032801045','3320019793','5100023258','3700078813','19056912358','6233887979','3507400570','3338390001','87145900321','1115605092','7023406416','7192175386','7056098191','60265218062','4150151945','4450098914',
		'3077201134','2460001756','4280011836','30521006927','2310001401','7592530694','4116760632','76211120625','7299928223','5210015644','30521041639','4116700847','1087668100352','2066200608','5150024540','1600015765','3500097295','3890001143','3100010093','30521004081',
		'4149709289','2190874331','3077205825','7040400008','3340060110','31254662644','4125543084','4440010270','4300020142','2066200095','1800012925','3800020143','930000007','3338360002','7940058708','1500098637','3500014173','4427608041','7040400005','75379200264',
		'7940006147','7340510134','1380044789','37600293754','70559901151','3700075536','7002200705','4300008860','5000096371','84886000050','7684000305','63641205028','75764561340','7265500124','5210015653','3281207003','5100014982','4300020076','4138740650','4149743468',
		'2400002192','4195317511','7680801068','7764430232','5100014981','4180050151','1004149713384','4900018025','9300009016','608002214175','980080315','4300020075','3270015847','70559901642','36382402346','3320018169','3000057409','4000049990','60265227895','2880014626',
		'1780018450','4427605616','3320035255','68476631731','74941788655','2898997195','5100014919','1380010115','1500004535','1077098106345','5210004620','3400005240','3600051584','5210003667','4440015600','71941078636','2150004209','1380010342','9020856782','63782210405',
		'7079690008','3100019663','3500098416','608002213765','61112322000','2260097352','3160402621','5480042355','76211188554','1450000190','2898910504','3076803578','7684052987','2460001088','8000051306','71860497383','5210000058','2800021730','63748009631','9990071319',
		'7940035120','4300020143','4190008904','1700015514','930009137','1600016060','88949700095','1800042688','7192134345','4300020026','70559901514','3100067086','60265241937','9020876845','1114110107','7146426080','3377610030','1600018875','4460032103','1111168446',
		'2880000001','4223872213','1740022320','4157011022','61112312000','1600013977','1650058168','31284353642','1600018146','4300007929','4293441083','7101206006','5210004989','2400034336','7007467699','3010011964','71860497758','3338360035','2898910069','4300020651',
		'4133303361','2073516159','2260097252','4148302918','3400001327','1958239181','1258779167','3000003790','1500091232','74759961913','7142901455','3320000135','5150024241','7116982405','4440016408','4000050887','4150100823','7121415640','85808900305','3270015846',
		'85110700300','7047013772','2240064045','1780017589','2113100033','3077205701','7940011802','2073516168','2200029056','3400006180','1500004943','7023406450','49000022039','7592538004','7121415431','5000057368','4116501738','4000051300','1111168633','60265219936',
		'3810100315','7265501113','5210001744','70559901271','1800000338','7056098172','1600014085','4148300255','4300008495','1258779297','4116700652','1500000535','7630172213','1862770320','4119641942','31031028313','63221001551','1780019027','4149711179','4369507112',
		'5150025195','3700071512','9020879648','2100006866','7056098468','2055924016','2400032320','4600081566','85000376645','1380044785','7119000692','30521004400','3800023452','4000051129','8000005503','1090031582','7116922706','1380010340','70935100026','2586659225',
		'7940076380','74239275719','7192117754','5200005155','4460032329','7007781278','7079660028','4142011025','3800026321','3700088041','7750700009','1087668100751','4650002277','1862710419','60265228856','3100062000','6414404734','87786900796','68476633041','1952155003',
		'7832207240','4000035391','3600054652','5460071629','71785410515','3400035046','7592530756','5200004228','4116705510','3000031180','6233878473','4670406330','3800017991','7064001556','1700011811','2120076414','7942622110','7047020036','3100067192','70559901538',
		'60502100210','4369507109','1116210217','4430012084','5929057322','79452220045','7047018759','41736001609','1500096010','1800044763','1084024313061','1530020097','7339067719','2550020042','1500005605','3000056732','7007455963','1007236839383','2898910284','72225256852',
		'1370022216','64786519500','1001123349204','2073511298','7321400614','3700055949','3000031697','36382412502','4119691111','4118810188','4179060032','5000096172','7045004891','1530001499','7680800584','2770004016','3160401718','5210000651','1004980022088','87218100064',
		'4164612410','3800013895','7056098173','85110700304','7020055445','608002214182','7940053576','5210054722','71631055154','7447111267','60265227839','70935100015','7648922175','980012402','71941053606','7484718326','70559901483','4157011069','3400022673','85269700132',
		'61328709877','7007781271','5210015647','4470006648','5100025304','1600042730','1920099427','1450050563','1500002777','6638000540','4180050169','2150004213','7940038641','2515500059','2200022586','1500007664','6638000541','2000012086','65827622222','7684000306',
		'1707713266','70935100011','1500000534','4133303256','7321400108','4480077055','7121415426','70559901338','7348413320','65724304606','3100070005','3600053481','7940064948','88810925308','47726004402','4138730758','5100025153','1500091040','63748000032','4149710211',
		'4125543081','7121415422','72225266123','1600041224','3760088622','5210007114','5210003529','84024314778','2240039377','4138741224','5150000693','61124739024','1600043895','7423533005','7350461152','4460032114','4800126504','1380019430','7940001397','930000106',
		'2113135024','7680800215','4149713757','1480064535','30025810935','88491211221','5210003004','80717671274','930009067','4175702220','2113100036','5210000444','1007236839380','65827620102','1600016422','3400050293','1780015495','3500051119','3377611130','70559901132',
		'4200016218','1500007662','19056912313','3338360003','2120041756','7040400010','1087668100750','3000056829','2880014646','4179000435','1500099759','3150671004','4600012308','5150001249','5200032016','76211128789','3800025104','78052631026','1600040761','4149707367',
		'2880000057','4118810186','4149743551','74239270116','5480042365','7940058719','4125543119','4149713764','81957301555','5210004640','2410070569','4133304261','2001700313575','1780018981','980080401','7340510308','72277623032','88462310244','5000057020','2420000981',
		'7940021628','5100001397','5100017548','2100008063','7114055268','74747900003','78099395080','7046200905','4119641076','75764561440','2970002150','5100028645','4133535368','7756700250','2550020630','3600049754','1500091217','4240001791','1600010612','85716100845',
		'2310010648','85000376663','7265500093','2240039372','70935100014','3340072116','74747900008','5210082790','63782206317','1070070810','1600048491','7130040036','3760066571','1380010170','4149713686','3700085933','4650013074','70559901647','36382402314','7630175250',
		'3800024663','5000065996','7240006617','5000096366','1600045798','18473900034','7453410870','3120023472','3600053477','1112000407','4180020130','3500098414','61124739532','2400024665','7255411205','7047020034','8000051486','2586659196','8000051309','5100024681',
		'82927476243','3700053437','60559210044','85168100811','7457003310','4116709832','7192144352','2073516151','7248600229','4149709491','1500000550','1600041221','2200028219','1800042793','1370077300','7045034391','2260000107','2400051062','4125583652','7255486393',
		'85000376646','4174000003','5210002245','84024314767','1780017587','3160402674','4133303831','5100027700','7209201210','2310014369','1500007668','3150000539','1007590000656','1111101490','2880029304','3760081538','2880027736','2073516137','3087744800020','30521233500',
		'7142901453','7114010100','4200035515','74759962458','7255420987','1700006837','4138711628','1005783617046','2586659224','2400016703','85808900334','9396611320','1254601213','4000003237','1500007493','88491212965','5000096019','35000076662','7253015772','3400031827',
		'3500045836','7484740000','73891233001','5100021224','4149744195','7130040076','7192103778','2016922150','4610001115','3500068048','7891102528','4650014384','2620000693','84024311253','4149743550','4460003022','1707713260','88885359013','3400027001','7350403001',
		'2420003872','2310013971','7321406675','1900021524','2700000211','2900002791','85716100846','1740014002','86452400013','7339061098','7192117578','4116700843','1004149744823','47726004400','3400038611','2310011813','3663207771','1001123349202','70935100025','7023400652',
		'1650059282','4300008514','5100024633','2260094732','4133535621','5000095004','4149713690','4119691418','7023406745','84945500001','2100005503','3760016899','4300005726','7484720716','4430005957','81829001938','85808900369','1630015314','3760005946','5150000094',
		'63810267889','3400044266','5100013736','5150000095','74759962274','2066200350','3760069884','7680800462','4149709431','3400053103','63748059034','3160402662','61124738605','3338360101','30521275000','3600050090','85808900313','4200087568','4410019076','4650000801',
		'4133500027','3400023956','5210000055','7339080036','1090074331','2100000754','2983900480','1077098106347','7570615100','71860497234','5100021591','81829001776','2113100017','70559901329','1500099640','60265243117','3040079257','7218056611','3010012803','4149705844',
		'1450000895','7114055151','60265241932','4149705837','60559210032','7130000132','7940056317','1800044759','36382405620','4369505631','18685200016','85347100410','4300097940','5000088586','1340951742','2898997500','4133500040','5210015650','2570070952','84024311257',
		'3140006307','4060003416','3320010015','5150000306','6798108793','7020054029','4293441090','4149743712','1500004829','81032601417','1800085516','19056912302','3140007908','608002214168','2770067902','7101207509','7684010014','2880014609','7192188539','6414402073',
		'7027223203','7066219502','3600050217','2580002629','4149719348','2550020167','70559901196','2791701890','30234089457','4138792985','3338365325','4650062248','9884304034','4293440869','1290000330','1920079838','2586659188','5210015743','4149713708','4100000273',
		'61328709879','70865600148','3800024740','7110021400','7103801215','3010012806','81305501219','5003150611818','7457035321','1380055266','7079640008','1650057413','81861702359','7348419403','5100020355','7040900408','30521507000','7007468145','80276333170','7920067438',
		'85808900310','8494550000','7570615101','4116706656','1089593900123','3400014400','3077209402','1300000113','2548400738','4440015468','1500007676','3800022366','3320019744','7920025121','5210007118','7114055205','3600054651','3800027744','980080314','74747940005',
		'1356230084','7047013777','6414404713','4850025676','71785410574','78260516154','84024314275','1111101412','4440015758','3077209403','4460032111','5100028419','3500046384','1089593900122','3980012210','4149709530','4100000864','85000376687','4150812289','2200023792',
		'1780019025','4531022920','1258778902','1500007364','2400033513','1481305282','3160401269','2113150912','4138753205','5100028046','78260516153','2420009415','3100012614','5200032012','5210074602','2570001132','2983900023','7007457455','1079376010212','7940020240',
		'7940022336','85402100815','80276308154','7248600242','7940007217','5000050290','3700075196','5000066002','3320000128','1500093519','3150663502','3500046026','2073516156','68142103612','2880013015','84945500003','5210094269','4149713556','1380012008','7114066010',
		'1500091215','3890014215','1500021017','5000010217','5210004630','5150024545','7127956449','930000550','7756700319','5150025542','1480051324','5210000493','7592530693','7684010081','2460001758','1530001465','60265218065','7287881348','1780018548','7239201616',
		'3507400287','88133401260','4300020404','60265218039','3400035050','4100016509','5210004641','7308011421','5210000462','4149707035','2420009424','4138723442','1480000692','19056952224','85641600001','3338390205','3087744800016','7681114230','4149743999','4116741240',
		'64034402198','30521004087','1780018026','3700093045','7064001564','7940076420','1800085517','5100028433','2791701950','2570070949','7321406144','71941075636','5210004650','3160402585','3500046385','7667705876','3100070004','3320019724','76211189531','73891202234',
		'1600045602','7057504022','1081097900197','82942500006','85000376662','2900002076','7209231212','4000055813','7248601072','4000056113','1500091214','2400052376','7261347336','4116761803','4110080810','4460031774','8660000925','5150014172','85420800503','85347100464',
		'81093403021','3800084561','1740011867','3485618603','5000096174','1500018673','7192100379','7764490045','85921300589','1500007650','8000049563','1004980022021','4149709573','2150004201','1258779278','81957301326','9396651300','60265229669','81590902087','4300020141',
		'7764490040','608002213901','3900008587','60265226961','7418245634','31284353466','3100067073','7920004903','2400024666','4119648674','4149705849','4119691116','3077208818','36382402328','4118811425','5210015648','4149713597','2150001340','81861702176','5003150620008',
		'5003150618108','1500007657','5020001329','88133401292','1380010321','4157005468','5210001384','1600010528','76211128785','7116971403','5100015872','4149709503','3760080795','30521004300','7116904049','4000001232','4440011250','84024311252','2400051066','3663207476',
		'60559210021','4116701760','1500007368','4440015358','88885300092','2400024523','4116514445','1087801560','4149709628','7007781280','3800024911','4149722375','61124739552','3160402843','7294075705','3360042010','70559901266','4610000694','2100061250','37726004403',
		'65922310210','1450000903','81097900357','7684067444','7079650040','1500002775','1090000383','84024311259','4116709372','2310011020','4369507107','1500007678','1450000897','4300005396','3040022273','2100032307','7007400365','3600050219','60265243116','3000057209',
		'7130000008','2850012065','81213002073','2586659200','5100021331','5210000716','3007144812711','5020056000','4146039993','7940044834','3100067024','2400057818','2052518365','4132100748','85808900314','5210000461','2052590124','4293440841','3000057584','1159665634',
		'60502129046','1370079347','7326000726','5210000657','7047016128','1254601214','31284353636','7766113783','3800026803','3400004096','4156500006','3700012784','3320019792','4150001095','84024314785','36382421366','80717671273','3160402743','1330076080','3150000319',
		'5200004823','1480000755','1159693038','19056952221','7478553424','5100027624','4100073149','3150610523','1356230016','7340510276','7321400528','2780005373','7726009644','3100067191','3800027223','4138741215','64420942564','3700059590','4600084351','3400071436',
		'7116904449','2850010265','5150002215','2073511227','7766110360','7766112307','3338390430','74840428794','3500068046','19056912479','7265500992','3133248254','4293441076','1500099789','5543761241','5210015742','3120001451','3150669304','89520300144','4149719347',
		'4116705840','84024314408','4114319270','5200032199','3800007012','7103801217','5000050294','2586659226','7119001706','7340510224','2400032200','7116928649','3100012613','3800016591','5210000212','7007457512','1007774527479','2819000679','4119691181','4175701863',
		'2400024423','2240000189','3400004019','1740010961','60265227896','1600018915','7007456975','2240065229','6798108926','5360010061','4114308956','7261345189','88949700052','3760039747','1800042696','7064002289','5150010837','3600053478','3980010291','1356230063',
		'1500007346','3160402844','4980022020','4650072442','5210015651','1159655102','5480042354','4116700602','6414400966','5000050067','608002213727','7116971400','64786519502','7101207531','5210000647','3338390204','5150024250','70559901342','1500007302','5150024549',
		'81021903081','1330079480','4293441139','6','1356230011','7340510164','1700006809','5200005022','5003150622508','4146020100','4800101187','5100027558','4430012531','81213002052','1007236839402','2240055498','5100027607','4116709565','5200032198','77098190603',
		'7764482622','4157007256','2400052579','8000000334','2880013036','4428412300','60265241933','1700015502','5543760142','4300007604','6414404317','4428442250','4610000692','5210015649','3291301946','2240062190','64420947107','4790030316','2900002392','7236839393',
		'85068711050','75733955555','3000031271','7007467768','1081829001151','4149709481','4240041620','5000024991','81829001816','70115913116','3700091781','1500091050','4369505632','7008503504','75045500010','1780019508','1084024311292','3010010024','1500046657','5100014877',
		'71575610004','1380019023','3040079323','7103800044','2240062392','4100011976','2586659194','60502150888','84024314070','5210004111','3980091157','1450000011','3360042060','7940024407','6414404756','70559901389','1500007367','4440013938','1004149705503','7287881349',
		'3320000144','4113700007','1700002934','2220029556','81861702273','4200016224','73891233000','7940086676','4119674073','4138711629','4300000784','5210003214','4440018100','5300005278','4150099292','60265226825','2400013211','4650073332','4200015952','68142403614',
		'2400025417','2880016475','7764430432','1330060133','8660000011','7674007066','85347100400','7484737716','1111101706','7726009796','3760048683','85269700148','7255442106','60265226286','5210000625','7684000304','4460060379','7116981236','2200022590','1600016445',
		'1650055615','7418245635','3400024521','7142901227','7684040005','1600015857','7064001243','7680800300','4195300276','1500004467','1700000946','11037416200','3400024991','2073516113','3600049697','1700011813','4132212652','5150000093','84024314768','3700012786',
		'1708287636','76211101055','5210000428','1081000615019','6414404303','5250008008','3000031793','5100027746','3010011960','4116717151','81829001761','4146039994','7101207529','1500099906','5210003491','84024314766','10374162013','3760066573','72368393879','3890011602',
		'7764430132','4112907763','5210002359','85110700307','85103500323','7326000724','1920098343','3600054672','930000551','3291301947','5480042356','3651425173','2880016476','2340005675','85475800103','3140006312','1700006815','1500007680','7910068984','3077207618',
		'5210002121','85110700303','7940067025','3800050037','86452400011','4173726801','2830000049','7271400799','1111168654','7265500125','4149719357','70115913112','7130040035','5000065978','1600047857','3077206122','8000051484','5100025034','6343589167','4280011846',
		'3338390409','7287843864','89000015010','5210083015','5000057399','1004149713431','4138766224','4300020073','6233887978','3980011099','1380044793','4460002031','1111163461','3890001462','4440012528','2073516160','4100011975','4119691157','1700012813','5000096560',
		'5210004974','2700061234','75045500011','4223830536','1500091052','7173000715','5000050124','2410031532','4116741206','7940052199','31284353635','1500011861','3077207628','2200028732','3800026153','930000085','1780019294','4430005463','3600054647','1004149713506',
		'4149743462','3160401717','3500068176','3338390407','4116700641','7007781118','3760018802','4149705848','4110080628','7667705877','5783600004','2400056669','1079233889404','7684000177','1740022326','1600016452','3080000071','3800019842','3700052054','7261373918',
		'7630123211','3000056398','2740080026','1500007052','2800094476','84024312254','5000028813','7119001705','70612903391','7910000815','1450000102','1340951520','9990012626','3890003060','75045500074','70612903392','7067273170','4180050139','7008503502','7144157046',
		'2055914016','3100000979','7218056625','4767733645','85808900307','70612903395','2898910506','7342053122','4133500173','85000376674','608002214199','7626540414','3160402517','3007144812710','7756700262','85068710026','3600049978','7704335802','7910060166','4174000005',
		'81957301323','4116705842','7940052242','2880065109','2066200002','7630172653','5210004840','3160402683','4440015458','64420942562','2340000797','4300003103','1004980022019','5210015646','4650071773','2620000689','78260516150','7116923830','5200032180','7680800650',
		'1159655221','75045500042','3338334630','1530001464','7079640004','5929057349','4335424266','75045500071','4530029904','3340045051','85269700153','3160402730','75045500068','4149743470','18685200145','5210000418','85347100474','7047016127','2340001602','4149709277',
		'2700042045','4149743505','4100022300','3338322231','65938900019','7040400027','65938900013','4149719349','2073511245','3338390404','7471408574','84024314780','5543763100','7114055240','7756722482','4157001590','60265241936','3760080796','1500004471','4100037205',
		'75045500065','2100004707','4119691022','2310013777','5210000443','4119610164','7130000600','3600053593','1450000906','2066200199','4133534290','2880060045','63441853380','3700075204','5210004228','7684000308','5000011078','2100000745','3620001800','5210004865',
		'3700035830','2016922289','81851201437','76687800053','3500098537','4149705839','1780017690','4149709627','4116708150','4116706620','4149713700','75045500070','5210076069','4600046602','2260022319','88399068121','4149713561','75045500058','4127102260','84024314795',
		'4119691408','7100700138','3890004198','2970004112','2850010010','4148304014','30234030277','4119611064','1356200045','85000376647','4790013011','2055924008','3320009471','4149709452','5210000451','70612903394','4149719346','84024310651','8000049564','7274597498',
		'7040400282','2260060103','3700080326','3800024546','3800027092','7047013779','1500007304','3160402778','7626540415','84024314794','7342053123','2850010030','5210003026','72225256019','4000052543','5210000293','5210002579','7684000303','84024314065','3160402518',
		'1312000484','1650056418','5210005087','31284356930','1600013976','3291313879','3400004911','2073511291','3320097499','4149710201','36382427265','1111168542','1600018486','88491224927','3600051582','1500007670','1600016903','7684000249','1500099682','7766104213',
		'5210000477','2370900403','18685200018','5210082789','4369520009','3400024400','3000031188','73891202239','7114055250','31284353185','4149719353','5100027621','4610000967','1370026358','7585600027','3077205815','75045500072','2460001194','7144170769','6798195229',
		'5210014259','4000057978','81957301172','4200094632','3980003283','1114110658','5210004810','83874102432','5210000049','7570615111','7150501728','81957301173','5000058600','3320009470','1370020016','1087801559','7829615029','3500098597','31254663789','4149708302',
		'4119691416','1500004589','85068710036','3980004546','3700066541','3760029198','84024312499','7121414009','69538900021','1708287628','2066200606','3500044994','74816261452','2200022576','81957301066','3700097343','3545773100','5000071218','4300006711','85103500324',
		'81363602188','84024311080','84223400249','1330076150','7150501330','1600016920','80702016086','2791701919','5210035188','7184005080','87489694088','4670409844','3980008010','4149713774','4116706651','3700057607','88399066124','7002200462','85210933130','3010012823',
		'7007468166','1600014571','5360010063','4000056568','7756700179','7680800873','7236839392','1380044832','1500099757','74759941005','4180050126','1920098013','81851201520','1004149713567','4610000272','4800000245','1500099316','7255442278','1330076050','7127957238',
		'88491234329','74239270296','36382459116','5210005090','3400035049','7940011575','74982614034','4920090499','72225250260','1600018695','7910077515','3400031526','3320014031','1530020096','3377604510','7265500092','4116709680','1500007455','7127956946','1500007661',
		'4240026715','3485610818','5000050261','81049303002','7681104442','36382401465','3160402953','4149707009','74747900156','5000050259','5000087742','4154879671','1500007593','1001123349213','5150014174','8554720020913','74759940434','1004480075050','8000010671','7648900431',
		'2780006760','4119691084','1750000502','1700011807','1650056862','930009362','7103801282','68142102653','85210933143','8000000673','3500046377','3500052578','5210003819','2100007942','1001397103002','7940035030','4149708304','85269700147','7726004413','7236839403',
		'4114312870','7891170400','3940001779','3160401870','81851201500','5000050128','4149705841','3800016768','71279020170','2001626419351','4240018904','84886004330','1500007328','78099395065','7592530135','5210003828','4116701109','5100015239','1500004942','3500098222',
		'5000065980','7321400115','7056098175','4119640472','75710701105','60265227129','60559210022','7116982433','3800026513','1087668101362','1500001252','7066242202','88885300118','1500004963','66440186096','3140008715','7681104522','81957301507','4900017419','3160402612',
		'3810100048','76687899942','5100025271','61300874042','4146039992','7130041022','7321400142','1600014089','2780006784','4300000789','4149713702','7940006663','2260094552','70559901314','2073516153','65938900015','1500004619','85921300591','5000050296','4149713634',
		'7570615122','7342053121','5210003825','1500091036','7119000673','4150001108','85402100838','1500007692','5150010861','3620001860','4119691407','82927453165','2310012119','47726004408','1450002291','84223400720','7116915895','1500007459','84024312037','7111815000',
		'4149743458','1600041225','88133401282','3291313765','4150108320','3600040532','5210015652','1500007690','4149743554','5100028146','1090023273','1500096015','5000050122','5210001745','4460001299','84024312126','1087668100800','1380053486','4173600028','3760048264',
		'7142909980','4111625500','4138741600','7064001237','2586659204','7066242203','88491200140','8634160416','4116709921','4428421600','2001626400602','7287878539','7007781276','5210054716','1500007618','4460032150','7103812001','7255460931','2260002915','3500026919',
		'7064002292','2310011650','6026300026','7248601004','4300097903','3320014971','1500007674','31031032503','1380010334','4650076948','1570020969','5210054734','7940044603','3160401486','3800023434','1258779169','4149705345','1356212891','4180050170','7457002310',
		'4149743978','76211128791','5210000269','3600053588','3700086222','2310010736','7047013776','3800019982','7258670115','85068710005','3980012996','4600012309','1780017591','7116915640','907001590987','4149708303','7778202989','84886000114','7321400171','1111101413',
		'1500091032','78260516106','9396625370','2084223400128','4133366155','4149708345','2073516154','9396651500','85000376671','4132221079','4116706639','3600054659','3160402754','81213002067','3485682268','4650002986','4129491017','1500007656','7253000008','5210000738',
		'1500099641','31284357122','2396418105','1500004965','3600045495','64420900416','1007774529888','4428413039','85798300433','60559210043','81042503091','930009289','5000032970','73221650365','85347100408','3620001802','1500049231','3160401338','4410070984','4150001107',
		'2880013016','5210000252','64034403160','2400025053','85556911060','1380004708','81668001031','930000045','1007144850422','5150004450','88133400988','7626510706','7020054035','60265227148','1005783642018','7457012219','7218063813','88399066340','1007774529887','5000017163',
		'4116741235','1116210170','74840428813','2120059846','2001626400786','85994700677','3160404313','4100000861','1740014082','7289101144','37726004405','4119691094','5113193681','3700001870','7674007041','7274562621','7079650002','85103500332','7184004934','88133401481',
		'32259100615','81829001854','7457096921','4410060986','1500093759','63441853381','5100028529','1258778686','3100001146','2885202891','2570071103','1159691020','7056098653','1312000469','5170078234','980057146','7332172535','5100028430','4000052901','1650058683',
		'1001397104000','84024312047','1356200052','4149708314','7680801031','83997700643','7130040075','9056953781','2400033503','1920084251','7457001310','4138761215','5250005006','1500093800','7120217210','1380044787','4149713687','3485684062','1500007653','4116705884',
		'3500099361','5000023806','4440018400','5000035603','5000047653','4650072961','4127102813','7570615123','3160403026','4116758057','5210015746','1111161334','7184004281','3700098330','7350461102','5000058016','1600041229','7184009012','5114140647','7418244571',
		'4460060151','5210003806','2100001699','75703795051','85420800508','85001758905','3160401416','7340510160','1600026590','7056098179','5200005021','7116922834','4149713753','7320289191','2900002597','7053804184','1600016894','4157005338','4610003790','68142102075',
		'4157005790','3500046393','81957301175','4124412756','84024314052','1500007003','3360042040','5000026588','7684022031','1600049615','7130040002','6798108912','88133401479','8000000519','1600017888','2120057235','74982600057','7287878540','3700089222','2400057499',
		'1600041218','3800013870','4440015640','7116922642','64786590403','2550020356','608002213758','65922314044','7110021300','1450000898','4133535921','87145900132','4149707900','87489694680','7274562622','2066200033','4119691085','3980003284','89880600284','4116705685',
		'2310012592','1330055538','3270012018','87489600750','4149709453','4149713743','5210005096','3500051105','5100022832','5210003065','4223837754','3700075260','4440013948','1650058694','7110000982','4142005487','3620043157','1500007021','5000017168','76211191784',
		'3800026125','7592530696','1450000904','7294075599','4142012817','4650070031','7184020034','1500099684','1007236839401','2561162071','4800101185','2066200037','7101207532','1700017010','4600011633','7680800460','4116709422','7271407221','4173601014','4369514827',
		'7057504034','7340510141','61278110186','3700086226','3160401290','1500091044','1004149713442','7045003211','5480042359','2570071362','74982657950','3150631250','4149708392','4300000635','7120210523','2880029300','4149743459','1085961000764','1450003156','2880013013',
		'81097900325','7239233036','1650058151','7146424040','1862770325','2515500053','1004149713392','7116981931','7726009094','7478029354','7056098466','4428420800','1500002745','4650071775','74747910014','6731200576','79852516101','78778020000','5210000884','3270012954',
		'7265500094','3210002016','4180020088','1500099902','1600028383','8298809184','1700007803','1380055813','4118300170','3100010091','815590290','1380057933','1500007654','1380091819','5210082794','980080319','2880013017','7100732350','4133303258','2113150661',
		'4149744119','3270012959','2400001553','3400002040','87489694682','1112000402','7321406160','7184004139','65938900017','3600054658','7103801284','3800016092','2240039366','85895900445','7119001702','4149709779','1380019003','81957301174','4293441127','5150000009',
		'7684000254','4149713312','4000052531','85402100839','2150001328','2744336320','76026352932','5200032015','4542559327','5003150669453','4470000867','88491227310','1500007682','930009529','7007468138','7940044877','4133536547','4149743471','1500007672','3900008104',
		'73221650501','7184010598','7684040007','3160401237','4154862618','3000057550','1340951598','36382401866','4149709764','7119076230','7047013773','8634160916','1084024310123','3760088204','5150010840','1005783637009','8500002989','980082013','1650056676','3160401358',
		'2586659205','5000057542','4300020144','19056930454','1159640425','4173720801','4118811232','1380014332','4116705835','1500096038','2500010133','85000376664','1700015526','3500026918','4157009418','4440015348','4149709516','6731200573','4850020505','4116709934',
		'7020055112','4116742444','2310013742','7032801046','4150180101','63221001531','7092044512','5210004983','4149713701','4920004550','3291313763','7056098176','2310013778','2791702271','75928360007','8634160816','5210002315','85899004019','4127102258','7130000174',
		'2000012199','7684007657','7403066375','2460001195','8000000674','4149743571','4149744169','3600049974','84886000265','3338390411','4149708305','7726009632','4149764105','2791701920','1003600053587','7119000671','2400025479','7236810706','7684000307','5150002529',
		'3400027021','3320070007','60418300905','2550030432','7000203080','72225266029','3700094721','5210000334','7192103339','1114110111','87145900136','6233889666','2150001402','63221002410','3500078294','7339008445','81851201436','89000012300','1290000700','3485610812',
		'1740014083','1862778632','5210004135','2240039376','1500004491','3500045461','81213002068','60265228876','5000096168','3500097209','84886004440','3320000281','7091903026','7684003674','4150153730','3890003533','7119012076','88810901001','7040900480','7764490067',
		'8434840001','4116758055','2400052388','65827620252','84024310638','7626540417','60265227160','85954700419','3485618602','4150150538','7080017952','7681114252','5100014879','81957301170','7209212121','1780017028','7348414425','7007781274','72277600000','7119022370',
		'9056919736','4767748423','9396611310','4149713775','3700058682','7032801047','77661158661','7585600056','2880029319','18473900239','81042503113','2791892223','4330161193','63748059024','3120022822','65827620295','3700075533','2586659193','1500004562','7703401148',
		'3600043541','1111168653','7106343815','4150895760','5000050292','4138722730','74982644326','5233633149','89880600285','1780015948','7339061197','85000376661','3485684068','4119674074','7184003614','3700091200','74239270157','5210000326','6233887977','4119645384',
		'85666300400','4146020101','7321400633','78260516110','4138710751','5929057592','7684000309','5210073568','5200032213','4410070985','5210000042','3800016766','88491233377','1003507406336','7045004591','4149708459','8000051590','85808900306','2100008110','5210000583',
		'7340510146','5000013769','68142102085','4157005982','7020055466','1500096023','3700059570','8660000031','30521022434','7431258188','3800024382','1500092825','7040400025','79452200176','70612900377','7023406615','1114110673','4119611185','2000012180','7756700178',
		'7110000052','4110082033','4173603028','7236839410','1600049467','85122000341','4300096726','6414400983','4110080581','4149743474','2791892123','7910022405','3400040012','7940020244','7130040027','67667000402','4119611663','7007468084','7340510805','3600051533',
		'84024313701','4114315160','5000040738','19064663008','7590000200','4150001016','7101206008','60559209316','5210003830','7756727637','76026352934','7630891051','71941074234','81824002660','3338380111','31284355519','8000051816','7570615103','4116710031','2240026518',
		'3160402727','85122000346','4300020435','6414464148','3800027018','2113100937','5400047521','2791701629','4293441349','81793900010','85808900312','1356230210','7626510700','7119060672','1600041231','7161180719','3500067154','7130040086','4125501530','1600041222',
		'7471408573','1370036804','4116706621','7633821432','3160401489','7910024148','1510002625','7064002294','82058167855','1111163794','4149743638','4149713637','1800085134','1340951519','1380010066','820581047528','4150001105','2400051071','7418245096','2900001620',
		'7910075819','4440012508','7046200629','2400032202','3010054093','3600048379','5210004680','7130000017','5210000046','4118810178','4173601028','5210003495','7570615110','3076857351502','7786500156','4129491016','3000057275','1007343500028','1500007651','82058162810',
		'7119022235','7116925805','7120211768','1780017830','7418245636','7940051180','4110080585','79452200184','5210002200','2816000001','81957301171','4149743552','7192184436','1780018350','1450000901','8000000672','4119691095','4800126540','1114087102','2400004722',
		'5220017105','793760107521','1740010505','5100027101','5150000017','9456206235','89161300023','2460001036','3160402842','1626419496','2001626400676','2800024618','2620011200','63748000040','7106343816','4786599535','4980000405','1870000027','1084024310532','1356261006',
		'1510002624','3100019607','1780017953','8660010250','3400004023','76163520300','7332104001','18473900104','4610000114','2113100936','4300001661','1380023260','5150002517','1360002022','7045074391','1600014646','2000012065','1004149713436','2400025239','5100021327',
		'4119612363','3700059080','4650003083','4149708311','2073511286','3600053479','74982600061','64420900404','3890004199','75703700002','83874102192','5170081053','6414404310','37726004409','7340510903','4650076947','7066207672','9518801090','1380083382','81851201164',
		'61300871982','1003150658919','3160402576','3760048724','2880014648','3010011298','81093403012','81957301319','4149713680','7626540413','2310014487','4280011845','81590902033','1159655212','1084945500002','5100023405','2100006236','7674007082','7007464253','1600014644',
		'7255420048','501072482751','2073516175','5210003790','4600048048','4113702441','1003150663119','18473900234','85103500322','3700087474','8167908345','3140007950','80702016082','1500096011','7453491000','7091903025','7940026102','2620011213','954243015','4149708457',
		'3800019840','4114309071','88491236979','82058167879','82058145991','2880027807','85475800100','2100017122','2460001085','7130000009','2570070809','1600014643','2000012974','3160403131','1500004609','5210000452','4650070498','1500092815','5210004660','60265241934',
		'2780006793','7940019374','1600043058','1500093834','3600051520','4157013019','7239201614','3150000316','31284353187','85103500331','7321400619','820581678586','7116922835','3000007210','7835471904','5113193599','7674007083','1004149713346','5210003956','3140007875',
		'4240021864','4146020249','7007467724','1380086647','75703751901','4149743512','1600026313','4850020498','8566300403','1600042899','7192123959','3600046934','1002561162078','81833601354','1340951626','7130040126','2420009780','82058167863','7339008460','5210004998',
		'4119611065','5210000214','4850020504','4440013928','60265241935','5210000571','5100007620','815590292','88491212974','84024314792','1111169115','2200024790','1708289389','820581904227','4149709786','4470002012','64174882555','7940034367','7684010004','1600026368',
		'3210002011','7829151712','3760031262','4300004998','7119000370','4149743463','2073511287','81851201521','4300008510','7682812681','7120200313','1650055435','2001626415295','6731200543','4119691112','78778077010','4157005234','5210000588','1650058735','5210003954',
		'7116970002','81957301483','7192156313','68142102082','7020055452','74759940872','2200028388','4119610163','4650071813','4660033432','7274597658','3810017993','1500007308','4138722131','7274597654','65784688524','7057504006','1380021450','3500098167','4530000540',
		'1600014097','7570615120','930000032','3760083200','3077209461','6414400215','8660070787','18685200003','1112016805','7047018121','94776','2791702270','5450026020','5450019146','3770032725','5100021231','1500099660','7910052004','89000012500','4149713632',
		'1064383100065','4119691403','75733922222','5210004625','41736030135','4119645281','4149744117','7047018712','7064040076','1650058762','4114302874','2130050122','3770032827','8660070777','81075701165','5100015247','71156520004','1500041892','4589300110','31284355854',
		'3400017613','2260008558','94816','7106343817','82058198115','1450000012','5000064498','1356211722','5150000008','7910082241','5210069313','5000073906','5210003639','2780007810','5210000124','1708287985','3100067193','3160402717','7047019135','4149743514',
		'4116703318','31284353465','78778020008','5200032555','61278110183','4300021363','3291313774','4149743496','8660016200','1380044836','7236800709','5210000676','4600013008','5220009792','7114000651','3400044553','3160402680','60502129918','2052598006','85000376673',
		'7339008462','7127920071','85382600358','5150002516','5210003025','85000180551','2550020351','9990010026','8500003337','36382420120','5100028042','7321406102','2001700313586','4149708306','82058147354','1007184020300','1500007036','82058167857','1500007329','7910052014',
		'3291311554','84024314784','7007781275','1500096027','4149705845','6026300032','85269700152','7130040124','3732312533','4790030394','7835471914','2850010225','4149743467','3500098540','82785400191','2570070668','4110058098','7592530142','72225227240','2740069402',
		'4149709504','7940006661','4110081125','7056098469','85315800400','1370010940','36382474016','5220001102','7023012724','7007462932','7091903073','2570071367','1074639518209','82058100876','2073516146','7940071601','2550030425','3160402788','7271400537','4600083251',
		'31284358633','1125108812','74759961827','4133500051','4149707036','2073516169','4149707928','60265242995','3760027308','7940011803','4173601011','3160401190','78774823013','2055914008','2880027748','2570000832','3700073973','7192185811','3500098240','82058100877',
		'7940019743','3338390426','5210004131','3600050129','7092011234','4600086121','7891318606','7032882053','85475800101','1330055284','82058147355','1380044791','4118815002','5210000236','2198510107','5100014799','1450000902','4000057980','2700000104','85999400608',
		'3150673111','8660075240','7064002293','3120022621','85347100439','2700069086','1500092811','5480042360','64034403430','69511940230','1380010039','7111714905','4149708348','7020055444','4133303535','1800042792','3160401274','5210004820','89880600286','7835471918',
		'4149713705','2780007252','3160404172','7940018183','5210000264','4149713619','4440018650','85999400606','4800001164','1064383100068','1356249901','1600041213','4660033431','1500005608','63221006133','7274563620','7008503500','4149709742','6233881465','4460032094',
		'3500099400','5000096017','5000050496','60265243009','954242583','3600051531','36382469024','2460001098','1005783629023','1500091034','4149713746','7340510309','5210000582','4138755143','1500049237','1500093523','19056950503','2570000831','7680800461','2198510083',
		'2900002016','1500007652','1114110340','3120022024','74747910001','60418300907','63221002620','2570070408','3600053359','85068711027','7829121926','2073516157','70559901295','7040400281','2073511296','7340510119','4470009675','5210000774','7321406616','3800025098',
		'4000051283','85392300245','3100010092','8660000736','3800017248','1085414700841','2052595330','3600046840','1111101640','5250005055','1481322485','72277600232','5200033875','18685200144','3100001147','7684000071','1650058736','5150000075','2400003472','1258779177',
		'7115650165','7756722480','4149713681','3077206555','4175702629','3000004760','1121000001','2100004720','4600028872','7681114050','78778060110','7120228534','84042611009','2000016361','7940018182','81162002216','2700000261','2198510002','1186311876','1800085133',
		'4149743553','3500046386','7572044547','2550020352','5210000596','3700089282','4118810039','3150633009','5210000047','820581982805','4300003094','2400003948','3270012960','4149743906','4149707007','1500096012','79376011886','5020001328','1004149713349','1003150673919',
		'3700082842','4650073343','4850002035','3338320030','1063362255585','5210003959','7590000523','5150000086','7261345118','3150000378','1708288603','60265241884','2198510108','5000096005','5210007113','7340510268','80276303083','5210000386','3160402673','3700086209',
		'7440500986','2570000399','7047565712','5210000208','7326000003','7940015246','2100006070','1090028130','4660033392','88399065300','1700011869','7045024891','4149713630','83609391082','7829131082','7321400403','4410070981','7790019281','1001123349220','69112488199',
		'2073516167','8000051770','2073516066','800000067','2066200284','63221002583','3150630909','1500004495','85392300255','4114304941','36382400820','5210035224','4145911111','5210082788','85210933131','3320014331','4175702643','5220017219','85392300225','1356249188',
		'1330076460','61124739924','64786597971','5000040244','4410070990','5220009797','2400034346','4790010110','3890003342','8660070756','18685200126','7287817228','4116502200','1800044762','3600053445','31284358634','7431255367','2100007933','3160411259','3077204536',
		'2620014051','3160402918','4149708458','4149743636','1500096016','954243373','1001186312454','4149743982','1650059127','4650051367','3160402732','3150661609','1330028003','7007468160','7271400550','1004132221182','1380010323','3400099665','3360043085','4149743633',
		'81042503089','2898910319','3320000546','5210000246','36382417777','5210004132','5100027856','70559901234','3160404295','3620043156','7091906722','5210000373','61124736474','5210000254','4113700537','60265227840','7047018119','4410070989','2100007944','2150000074',
		'4149743465','7339061201','5210044339','7920006055','5410000182','3800018049','7790065047','4000051308','5210002992','52200009795','1159692108','7120206429','3022304074','1087899621','82058100992','4000056033','18685200109','1380010201','5220009716','7100730679',
		'4138753215','3700099919','7116915662','4116701650','7684010118','1500098639','4132212644','7726009036','7332174523','81127100161','3160402675','4116709650','4300005888','2100008147','2700061286','1500096034','4110059393','7680800874','3890000946','4173600180',
		'306136','1001186312333','7093499616','1500002771','85954700446','4149743981','3900004520','1780017026','5210003531','7045016251','2240000986','85739400610','87145900137','4116704320','2500010015','2000019757','5210000243','3022304185','820581220372','85392300215',
		'5210062534','1113216864','1870000020','5233691421','4410070986','9582940012','4149713779','7127957227','2066200292','7091901779','81833601356','4110080604','85000180552','2000012140','1650058657','7040400248','2420006057','4132220272','8000051766','3080000189',
		'1356200137','7184004696','2310014554','7585600055','7756713226','7114066402','4146020099','1003507401062','1114011012','7726006087','3077209189','1800012225','5210004624','5210007122','4300005394','2570070589','7007462951','71156501121','1481305297','4200087591',
		'7045212319','1001186311918','5000019419','7940006662','7130048066','4300000050','3600051473','1380087685','83997700648','85347100412','1380013308','76211123603','3022304186','3077207621','8976348000','7592530735','88491234333','3663203968','69511929932','8634160616',
		'19064664101','4300000953','64786597970','5210000387','7007468655','2880010313','7940035299','2515505712','1840080072','1700012732','4180050140','2066200193','1370014878','1003507401242','3732312010','4149709307','3022304150','64034402200','7116970004','3077209210',
		'3500097216','7348414452','1330076040','7120214023','4149743583','8000049525','4060037799','1114110522','3160401160','5100028147','18473900329','1111100636','3160402731','7110021398','84024314416','5100024781','36382417783','5210000391','4149708424','5000066284',
		'1500055490','82058197310','4149708464','81829001937','2500013155','7047018118','6661360956','1500007903','3010012414','5200032552','3600051581','1500096026','71631016538','5210003038','5800031334','7750700007','1600018782','87489694065','63810267895','5783616822',
		'8974476549','6414408037','7910075811','4110057862','71631055147','5480042358','36382402066','5210057410','4000051278','5100017639','4144947367','4119611067','1003114204250','1920098015','2220044790','89161300038','82058197315','5150000084','3900000821','930000284',
		'7592530118','85739400617','2310013253','1114110138','4116708016','3150610577','3022304230','1380019020','1600041226','75703733745','3600053381','8660024011','85739400612','4149713500','3732311011','2240066841','3160401285','87145900138','2073511300','81851201531',
		'4133536550','4767739129','2100003934','7066242701','69511912048','5210000707','1003507401165','2073511232','81957301320','1500007666','5000050300','80276333126','7116900038','1003967737730','7020079003','4293441331','4116701706','3077201659','4116758053','3150671011',
		'1708288601','4149713770','5170020603','3377611144','2770067901','75710701301','820581678593','5210002768','1111103352','4300000743','8000000681','7020084183','1630015317','1005783614006','4133366160','3760081983','3810018002','5210000245','5210000242','5210004830',
		'3160401485','1650058684','2066200038','9955515171','5200032675','7920045290','1500092810','4138732483','4149705838','60265224990','7585648844','954243503','87489600128','85739400611','3500099360','4589308664','3377601203','3800020460','5210000593','4000052535',
		'1300000986','2100007332','7299925803','5210004780','7047013774','3150631009','3150653314','3010050560','2400016289','5233691202','3729791447','5210004864','76211108024','8660012338','3077207325','1480000709','89901300023','3291311549','81365201029','4149713777',
		'4149743504','81829001800','3600045591','5000096343','77098103109','1500096028','76738703409','2460001097','1003507406338','7002200460','3150656009','1004149713385','1700012736','7047019138','4300000046','1600017824','60418300908','1800085138','5170020623','31284352131',
		'4138790483','7940045740','4000001104','85556900307','3087744800019','3700082932','1600026316','7255435814','3160402725','6798196349','71514111357','4116700331','4138711167','7682812682','68142102252','85000197405','85290900368','18473900026','1090070622','88491201786',
		'4156528513','7726010019','7184004253','4600086101','7340510297','1111112341','7111812500','1500007686','79645119241','7047018711','4116741312','1600048982','88810925464','5000050491','7091902691','2400025028','7130080052','2400025238','8105701165','1600018995',
		'4150001201','4149709636','2310014558','71514111356','1356230261','19056950499','81042503111','10374420615','2052598094','1920098769','4800121379','2150005200','7910083409','18685200107','4118300516','4369578261','7261347335','4141945566','4138711117','79376012462',
		'7940085866','7530606611','84024314069','7116927948','7248600252','7116900037','2260099844','2073516180','1700012793','4154898807','4660033393','4430005468','89161300022','7091901871','63256500006','7119051701','1159685086','7471408572','3760086676','1121000715',
		'1500000908','62930712204','7228712311','7184020217','3700059594','7418244573','85290900347','1004082201751','2066200467','3100000982','4149708122','1112023003','7020051337','4149713706','1500017460','4149765992','71860414654','7116982416','4280011843','4767748397',
		'85657900231','4149743457','85657900232','5200004252','82058168422','2410010426','3800026079','3010057468','4127102810','5220003472','4600011635','85657900236','87668101325','7630618092','31254662651','5000045007','3010012657','1004149713449','7007467426','84026630000',
		'2880013026','5210000838','1481306038','4149725142','5200004325','5210054724','65784646874','3077209163','5020001912','5210082793','4149709637','5210003831','4300008048','5000050493','82058100994','7535511253','1003967737710','3500068782','3140006282','7726009626',
		'71514172928','7120200310','3500053058','3160403025','1186311874','4119644082','7490836026','3600054356','1500098632','2200025560','5000089427','19056930453','7184009112','5220020019','82927414802','5210000502','7008503604','1600040983','84353610183','3160401637',
		'5210003249','81957301303','5210004602','7786500257','8105701155','5783602104','4150001097','4767700051','1356261002','48070099901','1500007638','4000046426','62930712334','7255423324','82058106738','1500093836','1380017203','75733925003','2190845556','3160403187',
		'2066200006','1068833992293','18685200009','2200029449','81957301024','4330161194','1500099680','9990071167','3500097221','60265227156','1500007684','4149743555','5220001130','4171681547','1370038469','7192123500','5220017218','7184020212','2400024859','81957301553',
		'1600014641','3890003343','1500007634','2791702006','7891112485','69511960419','4157014296','1500011855','2400025236','1001123310200','4149743908','82927414915','3600018342','954203981','7910051511','1600016681','3760046548','2085001795613','30521019772','7910512832',
		'4116704430','7116904149','4600012313','5000050304','7616800098','4767748422','1600019457','4110081138','7116925231','7111820000','36382401565','2001700313584','7339060038','7630890994','3500067123','7590000203','2500005277','5003150672624','1600019746','4133500126',
		'1125108815','84024312247','18685200011','1600018884','82058197311','1600049554','7339063620','3700074971','4300004623','4149743582','2150005850','7120209460','4132101541','7056098782','7091903306','81162002215','1780019397','3160403194','9616227083','4149705334',
		'69899781033','4116717402','5210005002','980083027','8976346525','5000032798','7080017958','1500007636','8105701720','7088123401','85001795610','76026352956','1500007663','2001700313893','3160403047','5100002556','7192178696','3760081975','2073511302','1600044804',
		'5100028690','7040400006','7339087597','1380017420','85556911051','8000000680','3485682261','7940066675','85999400636','3150635411','2000012090','79452200183','3600051472','1116211203','7910052011','3150663509','88885300093','1500099315','5220017101','3770032832',
		'7045084891','62930712194','1004149713440','3320009293','82058103451','7127957086','9616227085','7091901872','5000096261','84886004315','3077209165','2200023773','7530605656','84024314789','4650002983','36382473016','6414400982','746167810068','4156528512','5210003023',
		'2073516165','4119691183','2000012146','80717671419','7680800872','4280046078','1003507401156','4157005039','7910092504','3700059560','2066200603','74239270130','2066200390','4149719358','3340072113','1380088128','8660000025','4149709639','7684000230','2770067903',
		'1700012724','1600045891','2220044791','7349015800','7326000004','4800018707','5210003669','7626500449','5000049400','4149739740','3500032029','2010000602','9451400013','85657900233','7130040001','89000012000','81042503109','8660070350','1800000317','3500098211',
		'7940030209','3600035641','4138700501','61124739693','62930712254','2100008031','88399068341','85954700468','4144947034','7228712310','1258722277','3600054654','7015788270','5113196827','86000188578','7116915896','3010011265','7940027210','81833601355','36382421570',
		'3800020010','4589310111','2200028027','1031254662183','1004149713565','9451400012','2340002196','3600053257','4154813530','2100007927','4600011634','85656400802','1500091028','4138722442','7184005081','84024314412','2073511233','3900010033','7332137734','1500054666',
		'7633838742','1600045682','1114110012','4600012312','1500007688','9070015909698','5220017100','3760088203','5220017102','3400014600','1111168557','72225250130','7127957084','7119002467','1450001790','5210000438','7490420213','5220017212','1700006810','2055911329',
		'4119610162','7066253002','3150631309','85000197406','85954700404','7577931110','7116982123','3077209493','7032801001','74840447105','5003150635420','81851201422','3077209209','72552609631','5210000261','1500091038','19056912300','4149725926','1700015532','2552609621',
		'7130040029','1007351441137','5000096309','7332104006','2515500035','7726009037','19056912317','1700031382','5020001641','1007274510275','1356200138','1450002514','3500078331','64034402014','76954612341','4767739128','4150864531','5210004670','3700059565','36382468916',
		'3500026921','4149708415','4149713762','4480071140','1500038507','501072492675','3000057301','2620001462','1500002803','2240000773','5000050348','1380016695','1356212887','5000058018','74982644313','4144947323','7339090316','67984410542','1600018459','4149744166',
		'81093403035','9003150640104','1380014677','1087899620','85347100413','82178341200','1840080073','4150885020','4116700609','4149707926','7056098651','7349015903','3890010981','84024314056','3800022276','7064001952','1780010038','86151800265','7007464248','5100027855',
		'4650002990','3077209480','3010094060','3600053471','6798108772','5000048255','2460001059','71941071234','7299925503','7088123620','4116741221','7002206740','18685200006','2700000189','5210057411','7940011886','71514151464','1111101208','70559901734','2073514501',
		'7778201841','7119071337','81097900525','1003114260135','5210002578','2460001198','7056098780','7040900441','2240000777','2500013406','5210004109','1004082234389','87145900139','84353610184','4149710977','4111620601','4149743980','3700087915','81851201424','85999400650',
		'5000054418','3700068236','3890001513','85000180561','7570615121','8000050528','5000024028','1500007350','5210003820','2001626407632','4157002970','60265218531','7007468654','71028233908','1840080070','7348419401','7590000527','7130040047','85666300408','2003400015141',
		'88810901009','4293441125','70559901507','3150610566','1380018807','2066200278','5210004303','1121000971','1003507401230','4470002268','4149708121','2240000770','4369515835','23700056368','7674007962','4470000053','81007048052','1600012795','2586659199','7120200606',
		'5200004326','1001123310201','81007048050','1700009173','5210004136','4149709762','7130040122','7630618079','5210003493','1330073050','3377611191','2575302819','1650053503','84886000256','4832101504','3700077196','3800021018','5220017201','3400035047','3000057371',
		'4149708378','1112000708','4149709640','2620000012','7418247370','3400048501','4157014467','3500045460','7116900039','81851201414','7088123310','3004747930007','4116735123','2073516142','2586659755','8660000099','19056930349','88885300121','4157011046','1500000785',
		'30521232600','9451400216','3500055190','88133401293','1700011857','8768401025','2240000778','73221630156','1500089660','746167810075','8705272347','64786596339','8000052086','3500099398','2300035405','7127957085','1125101213','3600051470','5220017104','4149708952',
		'81833601232','8000051592','4119642377','18473900022','1085650200647','1700006812','5210003496','7940001333','2120051315','82942500008','2073511303','746167810037','3010054436','4800001197','1650057960','2310014557','1780010049','7142901078','9990010012','36382401470',
		'9451441605','3600051532','3340072110','3160402678','610382875','76687800116','1920099621','7088123318','89348400088','81529400036','1111163464','1330055306','3400014680','4670404940','8660075251','5210000286','3100010455','3077209166','7841112261','1370042777',
		'4150153723','7340510270','80717671417','4149743466','67667000403','4157013033','7786500255','1114110119','3700052470','1450001793','87145900012','7064001950','5000062243','3150667301','4149719043','85000197407','1300079630','6414404730','7535511248','82058199510',
		'7458440020','81007048055','5210004144','1480000760','3700074961','85808900356','5100028726','3160402749','79852516098','7940045030','4110080798','7910003670','85001211716','5210003530','4149743588','7088121031','76954672347','7458405164','6414400109','1500000776',
		'81957301322','1800012388','3700059593','4175702325','1111102479','1600016455','81851201432','4173600011','3377611199','81957301389','2310013743','1600049185','88810925043','7680800917','3076867930','19056952233','2016922130','1650059381','5000050530','4149725141',
		'2200021262','4116735100','7778202505','5210004860','1600018784','4116700320','3000057274','4300007875','7116922703','1356247408','2580002249','7684036395','4149713766','4119640481','1920097181','820581496104','1600014904','14428412102','4300008494','5020001308',
		'4136418745','18685200002','79376012544','2780007243','1500099687','4142005532','7047000430','5210007127','4149713761','6731205005','4110058241','3400014670','1500049234','7349015801','4149743639','7349015900','5210007119','85000180548','3500046391','5215970083',
		'3500098602','85153600717','3600019308','84024313703','4116706412','2055911349','4149765985','4000055546','2920000213','7120200607','1125108813','4146020102','4149715135','4110058244','7585600069','2000012637','7301500173','4300000914','4149703267','7626510714',
		'1186311858','85666300413','1005783629024','7940063101','7490453443','5210004843','2001700313936','7940045824','1800013133','4149743641','5220001104','2198520052','4149713759','5000023824','79452221012','4149708456','2800045721','7056098169','5200004229','4300001907',
		'74759941865','2198520050','4144947387','18473900035','1500092821','7184020214','4133303265','2400083074','85153600739','4119641201','81484001146','3500067112','2240065158','5000057844','7116927947','5210000256','7940049594','6233888090','3700077273','3760005154',
		'7591900006','7192150503','3077203529','5220001112','1500099317','60699101344','4110081127','7192194419','85808900348','2340005890','5220009507','3160402741','5000050302','2198520053','60699101346','80381023501','7684010011','2198520051','4000058924','5800031333',
		'7940045970','4133303264','18685200142','1004149744828','1340912842','7835471936','5210069958','84223400721','5100007508','4149766005','7130040040','31284358635','3160403226','1780018354','3600001247','5000064465','3010050670','3160401617','5210000365','1600048542',
		'7680800023','3600054646','1700012744','4119611186','5220001106','64420900401','2880027738','67984410525','5210004987','3500046891','2000012568','1071156580004','2150000450','7348419402','3500067111','7778200469','2198520055','2100005441','5410001305','7079670003',
		'5220007605','64420902017','1001482112501','1450003173','7192169873','7116970003','7940006664','7841112105','80381023502','4149713678','8556921028','3700041965','1068833992295','2570071123','3600048313','3100067190','4149743640','3800018364','3800026129','2073511234',
		'3800029178','3004747930009','4000020319','4110081102','2198520060','18685200112','7007468157','4149765984','5170099668','1780017703','1380055331','5100028431','7040900229','82785400712','84024314051','3160402720','1600018781','4116741314','4300000894','7088121034',
		'1356230066','5210054742','7130040100','4149713747','3100001149','3081833601189','7490837004','5220017200','73221650360','1007020055468','72277600156','5220009513','7299919503','4150153724','5210003750','1650059642','2800075883','5210000353','3450015100','1650059283',
		'89348400033','1800012228','7007468604','1380062255','1700015504','3160402657','3160401170','4300028575','3600045491','2073511255','3800012901','4149708125','1111102583','7910052019','7047015111','7047044518','1500029825','2840072060','2000012124','3940022998',
		'4149708409','4149708120','85808900359','4149708315','88491235629','4149719046','2740046526','2460001086','3077209479','88397815584','4300020405','1008976360140','4280046085','88885300124','1114011042','7192187838','5220017103','2575302821','61278110182','65827621322',
		'3600054352','2073511266','2880027803','1070080428','7680800628','79452220023','7007456368','7456260120','8000000677','2150001043','2100004693','7161811626','2460001193','4369564090','1063782209306','7457075588','5000055602','1186312321','7119091135','4300000893',
		'7680801178','5210003494','5220017204','75379200231','76687850202','7684010012','7120200605','3320000282','4138731118','4767730123','7340510123','6382490994','1450000905','73052192633','4149711182','2850010904','5210000045','85290900354','1112410011','4589310494',
		'8705271279','7306080929','5220020022','5210000472','5020000178','60265224986','3600051343','7093499601','2800067893','88810925049','7940049593','3004747930005','3700097788','1009955508012','7047565774','1480000655','8105701402','1005783629022','4149705346','8105701750',
		'5220007610','2460001043','4133303263','80276307124','1600018891','3500067122','71941053412','77011830008','1650053505','77011830009','8259201093','5210000355','5210076067','3160401882','8768401113','6414400111','1450000896','5000066080','1356230065','2340006696',
		'19056950938','3760018390','7192175914','3700079466','1500099790','3450015179','7091903168','3150656019','3800027551','85475800111','3077209255','2800044379','4440012518','19056950939','88810901010','3500098492','9990071559','7265509004','5000028237','2200029450',
		'7127957237','82909103009','3760035442','8500002990','3320009142','1650051911','4149728006','5200004401','1708288812','4173603014','3320094231','7020079001','2310010329','8768400404','3160404046','2880027741','3160401496','1700006823','7120217220','3160401894',
		'4119640483','4149703372','63441853336','7418244572','8265766556','6661318106','8705212749','5380063110','2400025188','4670400008','2198520002','5200004453','73702500026','3150633409','7326000727','4149713529','4000058941','2198520021','5220007604','88949700096',
		'4149766165','85000197408','3320094232','88644952000','3800016749','76954622342','74982657947','3890001139','2055912345','1600049298','3800027339','3077209400','85402100817','501072482744','88810925487','85650200624','1700017014','2073516109','4149765997','1600018305',
		'85153600715','1650058703','7835471509','3400014830','4149709911','4470009686','1600018773','3077203154','81957301321','2800080113','79452200142','1780018341','3150000521','4300001665','1085650200648','1700026523','7585607862','4119612328','76738703420','3663207807',
		'80717671421','7592538006','4600013055','1600012929','4023266759','1380016635','71028232932','4023266758','4023266746','4149743589','4115600054','3800026363','1003114260155','76738703404','1258778438','7756700317','2100007943','5210005304','4149709912','4531051960',
		'2880014630','3500047013','4000058021','4157050001','5150000089','70559901340','8266650080','74941788228','1090000531','5000050349','85210933171','5200004128','4131131182','5220001128','63782209306','1380091387','3500099395','5150016948','81455802033','7114055101',
		'36382424220','7192146841','1380016661','1087489600942','4154810133','7626500208','7080017957','4800000264','7056098831','2000012691','4149718223','5210003952','4149743637','7626500209','82058106737','8768400407','7116925227','4069763005','4650073334','4957813331',
		'4460060044','7007468090','7066223011','1500002741','36382488432','7119000706','5210005000','7047017997','3400014110','5200004863','81833601358','5210003807','7577932003','4850020422','1002480048603','7002200475','4000057982','7074010413','2073516162','4149722376',
		'4138732217','7040400117','7047019137','79452220102','7056098174','5100002431','7024717811','7101206003','1380016607','2001700313935','71941077536','1450001794','4138732311','19056953779','7265509007','31284356359','2198520006','2420004494','3160401791','2819000759',
		'19056952228','3320094227','4149765994','2389628787','1500093522','1650053728','1450000008','7733053005','2586659157','7056098166','1600015227','2073511288','3600019305','4023226980','3160401780','7321400117','3600054943','60265227140','1006305417020','1600015164',
		'5000057842','4023226982','94759','3800025222','7891318697','5220017107','36382481991','5210000454','82785400714','3320010012','7482261063','2198520016','5100019880','1600017382','7726009680','8705271671','4149709765','1450001571','8730056007','1450001791',
		'5100002429','4149707045','7778201843','7119080055','7024718082','5150000120','1082058185428','2100007337','1006305417010','7756700219','4132220273','1600018574','1450000380','4148303817','3160402602','36382426268','31101740684','2529300449','1600017079','1500002743',
		'5200004404','763384721','5210000263','4149711307','76430229022','76954682348','5220009589','74616781025','1009955508013','2880027804','3500055364','5114134646','5410000065','6414400107','5210005001','7535511246','76211134085','4149708009','1186310975','30234097138',
		'3120022788','2198520022','4110056888','3010094036','1800089380','4138710245','7756700157','2010000600','5150010835','7482270659','8105701700','4149709910','7490420215','1480000706','1002480048638','1450002485','76026300012','7144170977','5000050257','1600018576',
		'8000052088','36382407246','3500098248','7920091579','85650200603','3338300153','36382405400','9451441742','4119612993','1111546974','7633847056','4149743590','80381023500','84024314852','71941072234','81957301611','7027200245','3160402509','3160404214','4138709116',
		'7065006025','2260060107','81003518203','5220017215','3120023468','7616800253','7726009092','3320097540','7326000011','4116735301','1500094746','10374420608','61300876131','81000344023','1500004419','2780007256','2580002890','19056912462','84886004067','81851201420',
		'2010000601','2780006561','4319262001','5220004102','3736300912','7066253001','4138731351','88644950817','7261347333','7119000687','1800000127','4116742441','71941075834','6233895499','5200004316','4470001990','7172004200','7940007021','3077207898','7490837002',
		'3000057300','4144947533','2260092661','7116925229','5210000350','7274500126','7218056507','1258779009','7093499605','2073516136','2198510045','4589310749','8000005550','7116915901','7920004905','1330055304','8660024001','4149713703','4767738309','7940019376',
		'85153600719','4149707069','4149707068','85210933114','4149718129','5220001129','3890004197','81455802061','5210000262','79852516320','2052518167','3160403127','2781517993','1114011013','4440012548','85954700408','1862711178','2460004213','2073516115','3800028185',
		'1380041784','73221650361','2898910281','8000051764','7841112106','69511960009','18685200005','2880027802','4119691036','1650057461','2007750700007','7020084184','7585600084','7064002134','69702911016','4116706654','4149725145','69702910936','1629144124','8000000684',
		'1330076426','3651424010','3700066224','2410010510','1077098103109','7294074749','1500098650','7047015112','4149727995','1862710226','85290900352','4300020056','76211138012','8500003456','4460031325','7756700300','1380047212','7910079538','1254601407','3160402437',
		'2000012110','7447100078','7756700320','3100012027','2800014891','5210000381','73221650368','3160442546','1650059286','6414431650','4150001091','79452200305','80717671341','1700009033','1113216862','4175701865','5210004973','7261316191','4957882364','3160404288',
		'1114011043','2010000603','3900004517','1700009262','2791701893','4650073333','5210000638','4149713763','4149709905','5150010864','7089622539','2200029452','4670400032','4650003478','1004069771310','2198520064','67984410454','4149727738','19056952232','7273069064',
		'19056952227','6798108737','3077203156','7047029062','7940008636','4082234272','1004149744852','4149713562','5220001131','7109160104','7940008781','5100021590','1650058763','3800026805','7940008391','5200004129','4149728005','7119076233','81441600190','82785400185',
		'4116741274','3010010895','3320097538','4116700923','1629144102','8660024064','5220017108','5000057814','81004816002','2066200142','7007468086','7064002126','1450002403','5000057531','81781001186','2900002700','1004149744849','5220009714','68476631741','2850010902',
		'76430290138','8266650090','78099395037','71514121007','8660070888','1380017419','4650001651','3160402434','4800001191','5210054733','4149718217','2198520020','1600049175','1380098306','4144947424','5210074604','5210052178','5210000424','7100731810','7940046095',
		'2198520097','2529300380','5783600052','3077209208','80276302857','7239225450','820581476175','62802501957','4119641933','2198520003','81833601182','1630015315','4800000195','1001342100120','5114139386','3600054237','78099395002','7940045814','4957813334','4023226981',
		'2100064618','3150669413','4000052545','76637579308','79645183641','5210015848','5210004850','85475800117','1600019707','8000051920','4149709204','2781517991','3500099664','1600015874','7490837001','18685200008','7940032722','82058106741','8265744707','1380016608',
		'7940035293','597015330','4480071150','2580002080','1450001792','7116925232','4149733368','7218056633','64786510030','1356210923','7910052000','7007458052','84024314781','3800028187','76637579242','2983970049','85402100847','4173728012','3500099675','76637579207',
		'4460002001','980012526','4180050127','75900003274','8768400405','7940049598','88491241831','1078142156305','5000050531','4800018706','3890003326','3320097535','501072483304','4767739130','4149733367','3800026977','3760017949','2781530727','1380065455','3600051345',
		'7756700243','3500046392','1500094745','81004816000','2113100935','7116932278','72225228808','1960004670','3500097293','4116700338','1356230228','8105701990','7482200006','1007184020302','81455802159','70559901393','79452200303','2100007930','4200087605','1380016619',
		'4142005219','2220044794','5150010841','18473900213','1380068332','3800028189','8768400397','8105701610','1078142152470','9451441741','3980013067','4300003164','85001758920','1630015364','85002726394','4149725700','4149725699','2240066936','4149713704','411100001',
		'7457040588','1186311006','4800001475','1480000232','2927451420','37726004407','3663207809','2240062229','4149766161','853693005193','7119030427','4149709964','85210933184','7756728323','3700071447','3480060080','76637579206','4149727990','3770032814','2898910288',
		'85000197412','7110021123','4149713771','89177000205','2073516177','1330020441','60265226561','4174000004','1600049392','7066242702','7092300300','4428412500','2005923050','31031032460','4149713991','2198520098','3400040611','1356211054','5210003829','3040079343',
		'3600054639','1340951755','1650058698','4164120103','8000049542','3400093788','1356200260','1085414700869','1600081331','3291323876','7458452110','4149709303','1600041001','74759961829','3736398663','1920089346','4319210500','4149727754','8500003164','7616800004',
		'2300035005','2880029361','1600018465','3500097217','1007126809100','2200021264','1700012797','8500003455','5210000043','72225250023','1510002570','2198510051','5210000780','75710701129','4110058964','60265242989','5100023318','3150673909','6827436028','4110001511',
		'5210004273','8000000528','1356249194','4119611072','1007184020301','7116925286','7091902103','7046200842','6798196302','4149725670','4650000329','3800023449','2084223400130','84223400717','4110080743','4149728003','69511940414','1007343500029','1004263600284','1114010489',
		'4149708350','3400037928','5000042582','2240000775','8000051918','86000133601','7478033355','7633838147','1085414700896','3160404212','8660075233','63256500012','1380044718','7066207671','4149719050','2389617854','2791891223','7109160106','1356261051','2620002104',
		'3680049727','4149728007','74941788327','5380063106','5210003532','68833992450','7080017951','1068833992721','3160404246','2410011667','2004175702572','18685200111','71785413101','2198520023','31284353188','1500096014','73221630155','7910052009','2700037892','81002354008',
		'76770700122','4149718229','4116704235','8700037924','4293440780','1085414700842','1600047738','7119043801','3087744800022','71785413051','2265571546','1600017991','7920082263','5200004923','33160414271','7623942120','2001700313932','1330060590','7829116916','5000066073',
		'3150662409','5000057057','7007457430','7940045971','1078142152430','4300029322','8266621812','7218056671','7116970005','3150000504','7910026635','81002354000','84024310535','3100012024','63256500011','3680044457','4149708460','1114110639','1085961000766','7172000769',
		'7089611623','7007468088','5000048994','1650058695','2190874325','2100002824','1862711176','3077200446','7047015414','3338365303','4149711856','4124461225','4149709963','7079690005','1700015529','36382405026','31284355507','7091903239','2310013530','5220017202',
		'5220009712','3770032332','2084223400131','1600048983','5320000324','5210005100','3890002905','1708288258','4920004754','7684000360','3700076359','3338370050','88885300122','4650004065','4116701712','72225228205','84026632458','3800025411','4149728057','1142300338',
		'76770700114','81007048053','4149727691','4460030046','2265571555','4263600283','3700052951','2781517990','5210096842','7321400658','2840069863','7091902383','3100067189','3400014530','3077207691','1650055531','5210054725','2260090126','2198520090','7490453445',
		'2840068386','8634117052','1500004493','1186310976','1600018753','7089656020','8312000414','1258779166','2310012294','89391900140','3680044456','5783602189','7161100315','6798108771','3160404143','7089665172','85290900353','7940048022','7210864001','2150097600',
		'1085638200365','3400056046','3760041823','81957301592','7423527341','1114110672','7116924985','4300003105','3600053382','4000058097','761328709877','1600029603','4149709770','1114110331','7456200111','2898910383','4149709785','4100068963','8768400511','5220004149',
		'31031032560','7500299924','2198510047','1500002747','8312001013','7274500108','3000065150','8660000020','4470001090','72225218724','3600051471','4480075255','4600011116','4149708417','1111102478','65595600012','60559210042','4100059300','5100026759','4508422301',
		'4850020378','7482200007','4149727991','2800032044','4660033391','1111102607','1629144184','5783642951','1500091046','2066200141','7431255271','19056953780','81363602022','3500047012','7457060785','81590902113','76687800054','1600049426','4200087597','1037442033',
		'2300035145','1600018575','1500004416','4610000712','60069900424','3160404210','36382426166','4149713452','3620001369','7274500128','2370005638','5220004104','2100004329','85159700777','2570000837','2190874329','68476600312','4800001119','3810012892','1708200993',
		'5220017227','71785415417','8105701420','4119612249','7047015215','19056952223','5000004380','1500099794','7940004037','7458405010','4157015467','7119000702','88491237750','4149706550','3480000641','4149725733','3600053469','4149707051','7007464719','7023010709',
		'7347200123','3663207329','4138722111','4149743569','7457044288','61300872178','7101208004','68833992290','2000012629','89177000212','18473900326','3087744800021','5100027098','4150000109','1330079590','7091902495','4610002020','5000058496','1500004588','4149708346',
		'4460031049','4138722151','5210000646','4149768026','4149766002','1380017132','1159604055','78302503010','4149733373','4980035300','31254600155','2791700123','2575303301','4175702576','8312001341','4154802685','5220017226','4157050012','7343500026','2310012291',
		'7236839384','1570011010','7585600070','6414431610','7116932251','1480064533','1500005606','2198520011','1600049726','4149703256','1600018752','88885300107','3800025208','1007778202737','4149708123','7240006009','8515527423','7261316106','8105701275','5220017211',
		'7116925228','4149709335','5210054712','2898910420','1600020221','5000049460','84024312035','4149703357','85245500505','1380010390','7274500073','71785413050','1500004438','4149713765','1007102231552','4149728004','5100022746','81833601184','80381023566','7681104050',
		'81000344026','7447101108','73221650510','2552609160','2410011575','85027300519','1650059643','1380049492','1800012927','2198520094','5220004105','2100077229','75166641645','2310014555','1708288253','4149710198','2580002256','7920077717','7046200841','4600011644',
		'7910077252','2370005475','3004747930006','8500003178','7778203059','3485622691','4175702542','84747300379','3077200662','4369506200','4300008493','7684010154','1380016686','4300001664','3800021763','1090003380','7045014591','8660000012','4148303819','2780006349',
		'4149768089','7265509006','7684058074','81004816003','7255425040','4369566882','42500006092','2300013214','4149708956','5000058574','2400016724','2310013235','5100028528','7684000066','4293441325','4100058833','2004175701837','5200005302','1530001446','4138753105',
		'4650003078','1500018677','7570615108','63221003401','8000049522','81851201415','3500014179','1500093757','1114110341','4133303534','3890003536','4149727750','4149708010','7091902381','4149709302','1700009216','3160402616','3338370200','19056952226','5170099676',
		'7130000011','7680800875','2586659356','36382426273','7056098188','4800018708','5100022744','3400072139','2260064215','4142074402','3600051469','76211138015','18685200128','2198520095','5543760245','6233899504','7940052804','8554110259','7110021080','5210004712',
		'5220009827','4300007802','4149719049','5150017665','64034404210','74759961828','1005783625004','1090000311','1629144105','5210063458','3760072283','7456241053','4149768083','1004508422301','597015230','3547290222','81833601188','1480000185','3140006304','2073511260',
		'4149727739','7130000601','3320000172','7091902172','67045270063','67045270082','4149711303','73583804310','4100068962','3810017663','1001182610022','68142102560','2260099853','1380099918','4300001666','1159602141','1112430019','81590902117','2791702134','7910512829',
		'85666300412','67045270042','7130040099','3160404273','81833601335','4149703160','19600517665','7255444802','7116924986','67045277045','3680008661','1650055552','7457065149','8660012231','1450001946','4149768060','67045270041','7057504026','4149725738','1629144147',
		'1650059309','4525511760','67045270034','3100062001','85402100829','3000056850','76738703419','31284356827','67045270036','1300000954','67045270085','5210000044','4154886409','3500097219','8266650100','7684000356','2460001987','1087668101325','67045270083','1650059128',
		'5210003839','81305501880','3800026876','67045277035','7079630008','8105701220','7047000439','3320009020','60069900454','1003967737720','7339008450','85373600554','5100025154','4116705530','3160404310','36382459424','2198510052','5003150672602','4149703273','85189300118',
		'85402100846','1254601419','85373600543','6602200346','3120001735','85373600508','4149711308','4149743907','2500013380','3150631245','4149727755','4149709725','8976348806','4110081123','62797501167','7940046223','8298805474','3800026397','4149713563','64420942563',
		'7339063625','81957301610','1600016887','80276307321','5783602278','75611058002','4149719044','4531053160','2240000759','1800042744','7101206009','18473900129','2420004455','31254663786','84024314064','4149711300','3800019856','62797501204','64420900015','2770067907',
		'1258779168','4190008879','4149727689','4149709726','3800025206','3150633045','87668100671','3000031599','81049303702','4600012374','930009366','62797501056','3600051462','25753032047','3700071483','4149719048','3320000253','88810925046','66903565014','7478037742',
		'3800027543','9003150641502','7091902693','85189300116','4082234273','1330076310','7091903310','5210004995','1330000235','7056098180','4470001992','85210933128','5410007489','3770038036','679508501014','3150630945','1600040991','3600050981','4149703119','3651419925',
		'64034402098','7684000382','72552609515','7091903321','3010010893','7940045031','4149703327','1450000420','3700071969','3500098504','815590291','66903565074','1004149713381','4149705025','3160404211','4427605610','8660024093','3967707214','3150631246','88885359069',
		'4124499982','1600018775','8105701870','1500004407','7210822801','4800013265','4149708955','3800026401','2260000005','2586659198','18685200108','1600043081','820581678531','1330060441','1792900303','5391522684911','3500099659','7756700315','71785415412','5100028879',
		'89000015005','18685200007','5000081940','1500004413','4149709688','4280012531','4600043165','3004747930004','66903565002','3480060095','19056952225','75468600112','1003651412265','5210007107','3077207686','4650004058','4149709724','5000050010','76687800118','66903565282',
		'7940020603','3160402613','7940034362','1600040992','4154861888','4113700287','66903565034','4149727994','1007020055445','85189300120','66903565252','66903565624','1780018725','8224201609','2570071287','3000032256','1650056870','4149766892','2389657706','7091902321',
		'4150151946','7116927951','3077207681','3320009553','2740000031','3160402761','4154866771','66903565302','3485678883','4149706876','7864210827','3160400221','1068833992292','88491211459','67045271015','5210000131','3140008730','1004508422302','1004508422314','3549389311',
		'3700065517','85514000217','10','1001186312316','4000058353','7590000593','5210000335','2898910387','5100028432','3680008660','7007468107','4149733440','88810911524','3400004021','1450001965','1380017219','7940044854','4149708321','7274500072','4149711305',
		'84024314071','7940008390','5210054719','1004508422300','1356249195','5210000150','76211138013','3700053590','62797501110','5100028725','4000001160','1113216856','81829001687','60069900455','66903565272','7066242703','3663202091','7235002057','1629144180','5210003753',
		'2460000050','1901480344','1356210126','4142006568','62797501027','84886004446','3150669511','4154845092','7864210825','62797501004','4000051285','36382485953','1450003174','2898910381','3004747930003','1004082234513','5220017228','5100000336','2100004685','89391900131',
		'79852516040','3160404309','1500000532','2260099965','4149728056','3500045109','4149709301','81590902115','7091902177','4149709589','1780017262','4175702540','3663204264','1003507448542','7146424480','3249','1111163855','7294075403','7219607250','4149727774',
		'85001795607','64786530004','5200032431','4154810663','4300005879','8500003451','1500005607','5434714210','2791701945','9007175501473','18685200061','7294074751','3251','3877883016','4149703287','84024314975','66903565262','4132218172','4149705364','66903565360',
		'1005783608011','61300872701','2480048601','3250','81005213000','71070840824','76430290141','1600081341','7172053944','84053012805','66903565110','85189300129','3700075239','3360400318','4149709967','9007175502579','85000376600','7680801177','1600017989','8554110008',
		'7585600068','81305501793','7500222048','1700003920','85189300132','2800094147','3077207331','1740022315','31101741005','3800024401','3800020276','1074203181','4149709684','9007175501827','85000376601','1600040997','7184020213','81004816001','7795859249','66903565222',
		'9007175501807','9007175502490','1007300786539','5410000150','4149765469','4149709685','7116927923','2000016163','7940084060','4149743592','88768667574','5210003834','81851201078','1600047779','66903565370','4149721627','7432930010','8312000415','66903565422','1500000906',
		'88768667550','70559901331','1650057716','3700019505','9002055911023','2392368001','1007300786538','3160404314','31284355565','4116706623','81833601233','7940037021','4149740774','5150024572','7192176908','9002055912023','2100002820','3400035048','81305501771','2073516108',
		'83113400050','7795869391','9004615856322','7146420080','501072492677','3800020023','88768667530','84024314058','7192494001','2880027800','2100004947','7057504010','78099395035','4800002335','1007300786542','5480042454','7116927946','4800100167','7007466439','3210005820',
		'81851201332','4149709681','79452220065','1862710227','85514000272','76954662346','3600053361','4149740771','4144947321','85539500702','2200028597','1004149713566','4132237768','5000029258','5210000656','7457097316','8000051970','1003836140312','4190008873','4149727752',
		'71631096186','27300707005','1600045603','3500045459','85514000270','7795869091','4149703316','2100002823','3000057176','2850010903','7116904899','4149706878','88133401267','4149713462','761328709879','1600040606','8500002991','66903565842','85392300258','27300123348',
		'4531036080','1111102717','5000065990','81007048054','69702938816','2400034616','4149721671','1600018783','36382427663','3877861016','66903565442','8768400395','3320009991','66903565023','3549389323','4732590851','66903565380','81833601359','9007175501706','2190840775',
		'2100007333','4149700431','4149721672','4149718232','7287888172','3377602520','4600047764','31031032024','608002213888','9007877040965','7096992144','7384905525','89901304325','7010001936','1600019975','2073511276','7116900301','9007384926400','8660024096','7114000376',
		'2970034147','8000000526','2100006235','3500026920','85245500503','4190008882','7940056187','2005934008','4175702605','1450001066','7940055100','79852516060','81719202018','7033062321','1500008005','61124738898','80717671323','3800024646','7835471240','1380028734',
		'4869240100','3077209108','7116924977','4149709305','8105701830','1380084863','62797501232','85002726330','4300000658','5170020640','4119612327','7756700298','71860497235','3320009330','9002730012521','7273069030','2198522028','7680800627','4149709678','4110081122',
		'81058903029','7114010200','85210933163','3400093958','66903565082','7210822803','1300000126','2190829102','66903565502','3600053614','3770038345','3320094226','66903565602','3700079088','61124740096','4150001233','2100007924','4149725569','85373600544','4149708448',
		'4150001093','1004082234488','7007467765','1360003832','85001211717','1063221003118','86054500031','2460001754','4149708371','5220004100','76430222405','73112400277','2580002289','85373600581','2310011416','7940045329','3500045540','61124739072','88768667541','1862711008',
		'7832232000','2548400005','7091902173','2970002153','81007048051','7832233000','8259263318','2200021265','66903565612','4149707015','7235000019','3077208275','2800032592','2073516119','7047029061','62797501091','76211128954','5210006340','4149708012','4180033700',
		'1600015873','7116925235','1007790100401','7570615112','9007175504704','1500007599','5210005138','5210003832','4149708322','81851201535','81143500221','8660024097','36382421571','2400050999','5150055011','1300052038','73188801660','4460031741','75589101504','61124735023',
		'2100002917','3600051344','77347936492','85994700611','3890004196','7172087000','7790053608','7756700122','5963538732','5210005139','88810905002','4508422302','7119000705','4149743593','81998101206','1800000420','7062239801','4369506202','3663202085','66903565402',
		'3770038348','3188801408','4133303257','79452200309','7464100607','7033065054','4149709267','1450011126','84223400247','5150080189','1111103368','4116760208','3160402794','3600053591','3500097218','1600045601','74747940010','4750002022','7516647006','4149703222',
		'7910052012','4508422314','4149719045','1001182610016','62797501005','4831302803','7470200107','4149721673','66507201066','4149713992','7002200749','2200027625','4149768300','4589310750','76687899945','1085638200357','3469580521','2548400739','1007020055452','81861702257',
		'4149709686','79452220078','1600041126','2548400012','2400011763','81124902021','13668056630','7790053606','3120001876','1450000009','81833601360','1500007491','4000057984','2700000262','3077206789','1101740821','47726004406','4149768088','3500067126','7255447052',
		'1500004432','3800013926','85514000298','7778203058','4450005293','1111101201','1007111794501','36382459516','94282','3077205814','4149703341','4300020475','2001700313931','5220017225','88885359068','1833701219','5210003786','81998101200','5210005308','2113150699',
		'70559901268','81263902589','7287817229','85004220369','4149727773','7146402214','85392300217','7756700297','7790100781','81201902492','2791700025','64440413775','7146426040','7116924962','3160402742','4615831393','3663207847','1356200018','1356210104','6731205007',
		'2548400010','2260095963','71755010788','81861702088','4100000426','8660075211','6027731','2389613894','1629144127','5210000217','1650055505','2548400705','1380018803','3800015111','1600018885','1300079810','4149743591','8500003234','2740000061','1787370512',
		'7591900180','4116700225','8105701350','64420942561','84747300412','3338310150','7457043400','7096900437','3150663505','88491234334','85024100884','4149705035','2340006690','1001800037255','3890073313','4589310754','7457065140','7795869041','4149767586','4293453328',
		'63256500005','72745668050','1002548400013','5210004288','8312001340','7116925225','7795869030','7146430050','1101740823','76687899946','85210933117','4300008859','1920099812','3140008732','7490836030','2515500060','2310011516','7591901120','1650058421','7841112108',
		'9007175502487','4149728016','5100020473','2073516164','4142003848','4149715131','2970033141','82058149522','89353600280','7093499603','8105701550','1075166677605','1111102608','7464100618','7120200303','7209800250','82415060116','4175702536','7033063346','4149709951',
		'7116932280','4650004070','2791700027','7172053293','4149714050','4149703519','8105701450','7795869013','5210014260','7940048631','2260095338','4149768059','7112196728','4116700385','3338304681','1629144146','4110080610','1003861630039','4149707335','4149708397',
		'1380058501','81851201323','4119612326','1007020055473','5007648900473','7112195704','4427605605','4831302802','82415048108','1370063467','7832231000','7795869014','1450001932','2260093951','7756700259','4650004072','4531040090','8660000991','4149725592','6414486805',
		'85245500501','8500003454','66903565802','3800024907','4200015965','4149708013','4149768074','980055215','4149709677','4149708124','85210933156','7321400659','3700096607','62797501003','1600029602','4149767865','5000080801','2400052375','1380086646','7940051170',
		'4149708977','7111703502','3081506600116','7347200119','7261373935','5260328368','7490453447','1380018808','7910092533','4133303266','7680800600','2400025189','5220009706','82415062312','7146427040','4149704646','1003022306306','76430229024','7294074750','1901480221',
		'1600040994','4149709701','85373600536','5480042350','71894000163','82429513674','3400093921','1002410550609','1002410558105','1700031346','1380049313','4149766512','2389617857','3077200401','4149700438','82058122034','4450033913','9647086006159','7623942036','3700074988',
		'67984410557','4154802497','4154863418','5480042531','4831302801','1500008015','5210027462','3160404300','3666906312','64034403478','1084945500016','3700075477','7940058760','9460712105','19056952230','9003150666304','85475800116','7056098168','7633838203','4149767593',
		'9460713013','1370047402','4149767584','4460031404','1002410559205','2550020204','2310013774','1002410559105','2400051069','7121415424','78099395030','7347200153','2800068650','3800025210','60418300906','1002410559505','78539700135','1780018344','7684000354','1002410559305',
		'1600018573','2830000425','1003022300463','3400001908','1600027485','4000058124','1002410558405','3160404349','2781502011','2500010039','3600051415','85027300515','4149767743','3400044712','84024312065','3338390110','1003022309735','4869200022','89903900203','36382421574',
		'2007700100304','3338367002','3000057704','4149743572','3700097082','3500099658','7274584489','5000096007','1003022305507','4149713465','4149728009','85000704254','2586659195','1450001989','7940045743','9460711235','4000099170','1003022306307','1002410558505','1007020055444',
		'7756700149','4149718224','3010051901','7629502219','3663204266','86589100015','5000018396','1480000187','2420005019','7161190491','4149700429','2550020433','73475601093','2548400006','7940045047','88949752539','4800002672','7490420211','62802511642','18613900051',
		'4600011643','3077209188','31101741009','3800022965','7111702353','3770033325','5000050529','8105701310','2073511265','4149709311','1087668100696','4149700434','2389657690','1600015445','4149708848','31254662645','1087668100752','81998101203','2420009729','3077206070',
		'8500003331','2548400815','5100027900','9460713225','618947560480','63221000334','3861636017','3890072062','3770038029','2340000865','2000010381','3663204267','4300020136','4149766866','3077205948','2265572352','82785400192','3800028710','7116922704','1500091042',
		'3600048985','30234030274','2198520026','81005213002','1600041269','7130000205','60426299208','81058903030','7092341816','85003170029','1650058701','4149719052','5100020248','7091903316','7089665362','7033065969','64420900415','7089642301','4300009161','5210004106',
		'2130014830','1829001947','4110059295','7457065477','4175702532','3663202084','7294075504','7007467330','8105701360','8105701611','4116700381','3861637840','4293441252','3077208171','3480060096','7116924991','36382499531','1007091901962','1007496059667','1780019303',
		'5210003010','7126229487','49000099005','4149718218','14583624085','4149707334','5210015520','7478044704','7535511239','4149707004','89901300021','7535511241','4149703255','7795846486','81998101209','4149713740','7161811510','980012528','1590000042','3400093864',
		'4149728931','4124412750','7274582431','7002200470','2100008043','7091903318','4149713468','8105701840','4149711588','64786590382','7684000073','81263902588','2310012248','5100027609','4149705321','4132100524','5210003833','76738703401','5100024632','1003836140310',
		'5220001135','4650000802','4110081124','4149721626','7274580471','930000553','3150660104','9998807198','4149708975','5210054720','3338345037','4149721628','8224201680','2898910128','7116922705','8000049523','3160403192','2070000319','4000058017','60491300003',
		'4300009302','60559200758','3485691084','88491241840','3004747930001','3338390109','60504939531','1330000012','7096900122','5210000706','4149700430','2100007925','1330055539','4280046626','4149740788','82785400189','4149793845','4149767691','8266650070','7633838135',
		'4116709363','3210005819','2570000654','3077205147','4149706121','2781558223','1078142117010','7091902457','82415062612','4149743717','1600014906','82429513673','8000051974','1480000657','5250008568','1007795869314','7005700190','4589307223','2100005259','1650059377',
		'3000057587','2113100034','7047020015','8768400598','81189202189','4300003097','4149700433','1084945500017','64786530112','7347200120','7630122300','1650058342','8105701400','4149727437','4800000066','1600018892','6057761','1700021731','4157014410','1480000768',
		'7726010017','3210020026','7116925226','1007701301065','7274580411','4149719047','4149768332','3861636037','618947560688','4149718225','5508600018','4149711304','4600012505','5000091476','63256500063','8000000527','7848787234','78099341269','7940058853','62930701534',
		'7096900102','7535511249','1650058344','9541100064','1007795853030','8105701640','2100008109','7684058076','81998101204','7624942086','94750','2000042727','7349015850','3600051342','4175702253','18685200110','1380085050','62930701504','82429513692','4110059139',
		'9396600954','4149728174','3980010281','7240000052','81189202880','1650058761','7274500874','78302503008','4149704648','1500007694','7274500246','5210003835','2073511299','8265744643','1002561162074','7347200101','2700052385','81189202789','86452400010','7940012196',
		'4430005852','1530020101','4149793848','4149715387','81263902408','4149709498','3450014449','4116700883','2550077457','82429513680','81263902591','5003150610515','4200094563','85961000001','85514000215','4149708716','7835471349','7175501181','5100024635','3010010801',
		'81668001045','2800042451','9451442892','4149719051','8105701650','8312000413','818947560626','4149708374','2548400703','4149753700','3600043478','3800025849','7111702701','4300009312','81143500215','4149743573','2310010838','3800026984','4615862512','5220017231',
		'7116932252','4149703238','1700018889','1600017777','3861637182','31722052101','3450015195','7940049592','7960607092','4116704905','1590013402','3469512262','4149713989','7940006675','7184004330','1330055507','3120022697','4149700426','1007795859291','7680801180',
		'81957301566','1600028342','2410011469','81263902590','5220004148','5100027829','82415060516','8660000005','4175702255','7111702702','7007468082','4149711315','7007468607','5000096263','7648945536','7347200117','83609391113','1780016565','3010012699','618947560534',
		'1600017946','4149753733','2073511258','4149703241','4110080602','3836140308','4149715111','4149703144','5100027913','61124738584','4180034800','1600046148','1629144139','7626540627','2100005487','3210008374','5200005154','3500046979','4149743577','4149718226',
		'7431255368','3890000642','1004137610232','1003114200067','4149768081','2198520024','4149727744','82415060416','4149708763','2240000192','4149753712','3450015119','820581569860','1600041227','62797501293','7684000393','81189202489','4110059368','2400050793','1700012801',
		'794878211292','4149708331','2400050792','7127956532','88491234943','7192170601','4149708366','1450000440','2480048638','3800019905','7007457508','7274500221','3700086233','1530020100','5800014070','31722071130','7170055401','7623942412','2130016137','3007790000121',
		'3500076668','5210004993','1330000009','1082058145018','80717671324','4300003126','4149708864','1007795853020','7299925703','4149703679','2781558224','4149725729','89903900206','5150099283','1380054042','1004980015014','85514000268','7091902453','7795869040','48250000000',
		'501072492668','5290900369','81829001961','2265530558','71514150121','63256500067','7110048003','7192496995','2400050785','1001123359216','4142001625','4116705746','73221630135','4010000286','2100061257','88588016967','4149728002','3010094050','2400050787','4110081163',
		'7283067114','7116970009','4149711564','1780018190','71785415414','3077204796','4149708492','3800011336','7089635352','7091902460','7228711611','4149768330','7274582423','7111702500','3480060002','3710055098','3077207912','5400047618','7096900016','818617025145',
		'1004508422959','84024312161','2580003010','7175504511','1007020055466','18685200027','1450001962','1800013166','2310012287','1450001578','3004747930016','81957301552','7047010375','3500098493','4149708300','4154838960','4149715385','2400050799','2055911307','4100000207',
		'2420009726','4149709710','4138722792','4149703286','7347200185','3007790100651','4149713738','2756802825','1005783629090','1004263602078','7273069054','1002056911312','3150684710','85003170031','1113217113','74349000015','3338311964','7261373882','78778060048','7096900410',
		'4142010025','3663202753','4615851252','4149708360','4149708301','18685200129','3600051406','81058903034','4149713772','1862799948','1380017978','1600035792','78099395001','84024314870','980012527','4149713773','87744800015','86000002391','4149715120','1380018805',
		'14583624092','7616800356','7110048009','1380088622','4600012357','5003150610415','2700042071','7878677777','77347931056','4149768301','5210000258','4149728011','1480000422','3320001110','7274580401','5380063105','1450001747','82058101859','88491237699','4149793853',
		'1480000750','81851201530','2310011415','4110081099','7629502245','2410010592','1356200061','65595600954','2200021267','73221630182','1450002159','4300005377','77347931053','4110058990','81263902395','4164110101','4589307227','7067275150','4149767608','4149708976',
		'2400050791','81957301545','31722052001','81957301325','1002480048602','3000057703','81124902027','4149708376','4149700425','8888888888888888','4149708858','70969002469','4149708341','4615830428','36382471016','2548400670','7127957087','4149703221','7283067119','4149725568',
		'89391900133','4149719401','3000031600','81735001002','3967787398','86000002393','4149753702','60426200308','89353600270','1780018956','7274508645','7116924961','31031043001','1600017517','85441700218','4149768020','2880027801','7535511261','81189202589','4600081181',
		'81829001834','4149743515','5100025158','1111101642','3450015184','1590006337','76770700118','7161100307','2791700386','5000015815','4149708347','8500003235','2260000153','4149766867','2310011021','85514000299','3760072280','1076857381002','1650059580','1600045604',
		'7192181007','7111702505','4110059415','81565200421','85347100462','3450014002','7648951588','7349108140','3400072138','1600019370','4132213760','1330000251','4869200048','5003150633714','8439506101','1380016651','7274581461','7756700318','5210004601','61300872176',
		'4149703264','4110080521','1450002378','1076857381005','31722070530','7629502229','7447110052','1069511929932','89903900200','7116925223','4149715449','4149768335','7940020243','1007795859236','3320097240','2400050786','8105701937','2260064237','76430220421','7116932253',
		'7535511523','4149743568','3700094860','64420902142','85282300608','3760083208','2055914022','4149753006','7116925243','4460032075','1500096035','7684000386','1780019301','7116925224','2400000838','62802511641','4175702251','5210002335','2791726009','4149768327',
		'4149704651','4650076749','4650004061','81957301648','5210003826','1708200803','7778203057','4149728014','4149708296','7726009621','4116705561','7535511235','80276316864','4149767597','89903900273','3450015129','81833601334','1920098390','3600045599','4119612137',
		'1085414700892','82429513641','1007701301021','2840070002','1370092330','5220007619','5210004280','3469512286','4149711302','3100011478','79316592196','4200094569','4149708764','4615820427','4110058929','7111714426','1007795853023','107701301025','1629144185','4149728055',
		'77347931020','7089622872','88133400678','4411512840','5170020639','7274580700','1740022324','1076857381001','2260070008','4116701071','4610002023','4149709811','2400024660','49000099169','7349015851','3400093861','7056098778','4148303810','4149707566','4149708298',
		'7680800652','8500002900','4149713994','81590902114','64286311052','84053012807','4149708707','5210004607','2100000768','61300872222','81833601234','701604126959','5391522684928','3700054140','81050900910','7005745056','1500093571','4149709263','81829001684','7111702514',
		'4615820251','7283067123','4000058943','8105702032','3077201570','64440413705','7255422576','2840069867','73052150843','4149708849','1600018978','4149702753','7130040044','3666916186','2100007923','85003170028','60265243266','4149767690','60699101478','2400052444',
		'1370021651','1600018466','2265530605','81058903033','4149753731','84551401112','4610020297','9541100063','85961000620','8105701745','3620036586','1600040920','5260328367','4149715448','85210933179','5210004006','1002480048622','7255422341','7126229492','2240000774',
		'84026632459','7064002146','3663202782','4615851254','60699101476','7940048686','7535511521','7020052344','2898910252','1480000748','7321470948','7829151820','4148303825','3120001263','5210054721','85210000700','1007111703034','81189202858','2400025029','1101757136',
		'4149753171','1600016365','7766116206','7684000235','77098130100','4615851251','4149719400','4149768397','2700037998','4149708367','4149708974','7349120100','77661136768','4149725667','7283067109','85210933142','31722015130','85308600808','1380037961','76172095054',
		'5000090742','2001450000144','1115605095','4900023487','4615862513','2840072062','85000197411','64420940820','4508422300','7680000013','7591902001','4149743517','60426273807','3600018339','4149768232','4000058007','608002214069','2220094154','7795869053','7144170975',
		'2898910417','81957301304','89391300494','71785415415','7092350516','4980005721','4149715114','1600018974','1300000127','7274580477','81998101205','2016922252','85210933186','3800019879','4149767745','81829001803','7076770701223','1480058226','85282300600','1600016051',
		'4149753165','8105701880','7630175380','3800012569','89903900208','8660000090','1380067703','4149743576','8265798441','3000056167','2780006680','4149725737','85564300604','3500066053','4300022801','60559210020','4149703298','81851201837','7623942030','5220017233',
		'1650059460','85441700216','1089520300176','1004263604357','4116704365','4300006955','85103500362','4149708299','4149721632','89903900201','1007701301090','2970035101','1380010585','70405170103','4149727745','1380098364','70160419512','5000054500','3600037143','7766102912',
		'4149753737','3500099670','4300020052','7940048678','7630184002','7287888170','4615820437','7349109300','7161100309','4600010147','4800101595','1629144170','4110080955','71941078516','1007080006787','82058167874','4149767609','3890011469','4149702756','4082201246',
		'1600018468','88588017012','81829001944','7626598973','7144830019','60491300214','85001027209','4615820206','1004149744850','7940049599','4149768364','1007080074836','64034406592','4149713741','89520300148','2260097327','4149753141','2733100609','60505017009','60491300101',
		'2100008267','7007466441','2515500076','3000056854','4149703798','842644021856','4175702534','89353600262','61300874248','8500003217','36382451067','4149702845','7940064676','1740022261','19056912461','1450098789','7535511260','4149728024','3800027657','4149703244',
		'3000057260','4163807026','14583624191','4116701081','1380088212','2389680027','7064002303','1380016668','8000000682','7940048713','9990023974','64034406594','7096900440','4149713461','3800019619','76211133454','3400093858','4149709707','4149721003','4149708339',
		'9451442916','36382421467','1330000242','4154829573','3663202079','4149754312','5210001746','85210933124','3663207044','7347200141','1600020101','871091205621','2001626401100','64034406584','1087668100671','7007080021635','62797501277','4110058237','4149703199','7294075501',
		'2791700119','2310011418','5260328366','4149708364','81957301508','81735001000','7940048232','1004149744851','7126229490','2300011763','2400057742','7960691015','1500036601','4149709950','7076770701222','2260009263','4149708852','4639960024','60491300213','7349109500',
		'4163807031','7829170927','4133366112','4149768076','3150659503','85047700527','4460060176','3469580121','71941076366','7940045611','4118300512','1700017546','7116927949','6233802700','1111100886','9998807196','57237022501','4980016017','4149703383','5000022868',
		'3600051589','7623942084','85047700528','71631016537','7111702513','7680000012','7040900434','4127102269','4149718215','1001182610023','78302501003','4149713760','4450031297','7832257800','1650059096','659710054','2733100610','1600015444','4180035700','1600018708',
		'85210933106','89903900257','7056098652','1330065330','3320097498','7629502243','2400050788','4330161200','7027501291','3077206790','69706852027','1300000144','76430201803','76211130193','7144830085','7192162679','4148303821','3320006513','8660000008','7321400536',
		'2733100217','81998101201','81093403019','7648948052','4139000107','3077205630','4300000795','7079640010','4149753057','4149702847','2529300228','3680010033','85003170032','7101210702','3450044692','2700000837','7096900111','4149719402','73435096047','1600017272',
		'7283067111','2200026076','85898200139','1700021715','4149703471','82946200120','7535511238','1005783614021','1862710838','5000043161','81833601373','3469512385','1356236342','1480000186','84024314059','4149713463','4149702848','5508600002','7175504640','1590000060',
		'1380082528','4149766905','7623942777','81003518031','72368393848','4148303815','4149768235','3600054496','4149704645','4149753852','1007111702126','4800001181','5210005413','2700000001','7130000010','85245500545','7116925230','7347200124','7240000018','4369562899',
		'7580598287','4149709768','4069763004','2340005708','8600382091','64286311055','9073621125023','3800028139','4149703325','7274504128','2480048603','8660000027','5000018493','2400055117','1330055502','2265530604','31722071030','4149753045','4149708127','1007111702125',
		'3680000108','8500003100','85001257191','4149718207','4300003294','81005101048','4149740792','7940045347','3770034062','76430222404','2400050771','64034406582','4650004069','4149714120','4149702833','4767748342','81124902045','6731205070','7101210549','85493400720',
		'5508600001','1087668101422','1115606034','1600044492','5100026814','3120000231','7096095163','9073621125261','7726009623','4149708388','7535511262','7228717570','7116924958','69706852011','9933107907281','3160403011','5000050011','100134210006','81005101051','1262373373',
		'36382499515','9000272001822','2260000142','2310014199','1380019086','2130014111','76770700138','3076857351505','1005783642622','3004747930015','6076857350515','6076857350505','85514000216','3007962100001','4650000271','2400025190','6076857350506','4149765457','7864210828','1600045273',
		'4149709960','60505083305','31722070490','4132222469','4149715378','4600028739','3150651604','3076857351503','1064220556502','3320097241','1800000164','88949752499','6076857350524','76770701422','1005783642595','3076857350519','5210009160','2370005469','64034406580','7142909950',
		'77098130102','7045900909','6076857350518','1007020053416','8158401329','4149708401','6076857350502','3076857350503','1600018642','4149741924','19506929585','7020051401','72184442119','4149711545','3077210033','3076857350510','4767748387','4149718469','2500013414','77347936493',
		'85000446011','3800034894','81005033057','2260060110','3076857356001','3700048089','1800012324','82429513681','3076857350516','6076857350507','1002480048615','1360003119','1064220554030','7756700210','4149715126','19056929584','7007080074838','87668101422','7146430252','1113216866',
		'4149702652','4164170030','7633838810','7940049018','3469512404','3077210032','4100054100','7239610001','60265228364','7891318598','8105701440','8500003452','4149767412','3890000641','2100001638','4119612991','4149702754','85047700529','4100059292','4149700421',
		'3100046004','18473900349','1081060702071','2733102028','1005783642914','2550020188','4149702846','7126229485','5220017232','1076857381010','3400050284','60505267109','4164110103','85003170030','4149701262','69702922016','3400022300','4149713776','7283067125','2880029405',
		'3000057702','4154802385','80276312328','4149703537','4150001193','31031032270','4650000805','88133400967','85514000214','7680801160','1007496052829','81003518030','2006650800','7175501062','85000968204','7047020653','2756806034','7116924959','3010010227','57237000830',
		'7027200244','7633833570','7347200131','7283067105','1650058697','1590014063','4100058786','7349151000','3700071543','4136408486','7020053007','3700090245','4300001668','5210004243','4149705158','4149767613','4600082361','8312001010','7283067116','4110059789',
		'3663207626','69702923678','2898910512','7490835002','4149709962','1330060120','4149702973','820581466633','4600011115','1650059581','73114933837','3680008665','3160402682','75703700087','1600018189','6233877917','1500014004','5000096367','3507401102','1500004429',
		'3469580021','4149709961','3500098399','1380093835','4300009110','3549301271','7321400537','81829001686','5410000537','7116925239','7630618031','2733100603','1082058184991','48070000001','89245300105','8500003122','4149767610','1113216860','4149709982','85153600714',
		'4110081119','4000052533','5220001137','2340005697','3320009990','4149702971','4149728185','1708287756','7348419414','4800101593','82429513663','70559901220','4149768329','1064220556501','4154817492','3600054944','7057504083','4157014466','85308600801','7940046170',
		'1708201043','7484773132','7274504108','77661004449','7146428080','4149733441','7096095025','4149719405','7535511240','4175702530','74338007406','1002480048600','76770700127','2310011027','7756725444','1450001795','1064220554028','4149713778','4149727557','64786590052',
		'2150001042','597007541','7681134652','597014030','7294075200','4149714391','5200003962','36382421572','2830003032','85282300604','72225219323','7283067104','1007795853004','3700071477','4110081132','2515500068','7848787235','4149767594','1064220554029','1380082280',
		'4149736607','6305417030','85778600817','31722052201','70559901194','85210000732','7940049597','1004000001160','1870000070','7680801179','8660075252','70463999002','85001694418','7484718313','88831300031','83113400010','7490453439','7020052200','5100027104','1650059650',
		'3450014429','3600047918','78302502010','3077207255','42500008570','57237000930','85001694402','7187154860','1600010720','1629144121','7047020016','3450019406','2100030192','2198520025','8224201604','4149793710','1007020053422','31722071630','842644021870','4149718202',
		'4149753043','5100007626','1081060702069','4149705176','4149767620','77347956010','80276316865','3800019907','9007175501411','7447128996','3800027525','7431272900','7007462599','4767748372','81441600191','7170032561','2733100602','4110059788','1007080041061','85002007801',
		'61124739841','4149702969','3400024987','7096095055','1800000159','82785400195','3160402848','4149767746','81050901153','85808900354','1064041051327','3450015118','1600020765','4149715447','2550020408','4149713464','3700074962','7175501103','3800020021','4149767617',
		'7020085716','1380093594','4100000239','3680047186','7146401893','1600018774','60265228352','1780018954','78302505008','89353600265','4116703360','85003170034','5000046730','81005101068','81038702216','4149708014','5100027832','88133401649','7778202393','1064041051451',
		'3600053088','7032801042','4154890709','7170053001','1450001666','81005101052','1780045007','7119009819','4610035471','85210000744','89391900171','7940049601','64767100014','4149727560','3077102637','67331609462','4149703250','2400025198','7535511237','7550010004',
		'3338300703','4149727999','7940058851','84667500395','7007468070','4149743518','3150671104','7274504126','1356211991','7684000392','3700089674','4149708011','1780018951','4149711313','3077210040','64767100015','7020052202','1356212183','63221001020','7684010134',
		'74759962453','5000090339','4280012179','82942500042','1600017006','81038702141','4149703536','84667500392','5260328388','4800101594','7591930030','7274508808','3000057574','4149715129','7962101391','4200044472','1078142152335','4148304386','7175501198','4116700215',
		'4149754316','4149767686','81005101053','2700004204','3077210048','4119612145','5220007618','1530020042','3600042493','7116924960','4149708307','71941074006','4000058034','4149767619','8500003192','3680004255','5150078357','1700020930','8130801164','7349120200',
		'81735001001','2780006800','7112161067','2100008046','1082785400765','3500055676','4369584329','3010010524','4116709932','2052597413','4300000960','4149768361','7910022019','1005783642064','4149701263','1707713271','1085650200685','81829001960','1007020053428','5210005275',
		'1500096025','4149767932','1380066156','85564300608','81093403023','64034403162','4149718462','5000048605','820581678654','1003022305705','81861702260','7294075505','3500046771','7033065609','85001694404','4149714049','3500044988','81003518330','7005735070','7841112321',
		'4973309016','4175702615','2052510028','4149709984','51241','8105701012','7347200126','85509900759','31722015390','7142901472','7680000011','5210001337','2800057954','8105701180','7126229494','85000704216','4149766877','81038702217','57237004101','7274504073',
		'81957301665','2400025045','3700020826','1600016194','1600020075','84747300391','4149768347','7910092531','4149708406','5210004107','3010012368','7005736559','85785200208','4110059452','8051613510','4427602297','4149708878','81058903173','60265219912','81124902040',
		'7175504643','64767100006','219852202','7339063621','4149701052','2260019114','7080074380','7432930014','7192112149','4149768226','7274504072','88810905004','1003507401243','61124739073','6731205064','3120001453','69706852019','4615920201','4149727558','7283067126',
		'4190008876','2410011690','4650003808','1629144114','13668000701','2007080082281','8000052090','2970035100','1001099500842','7756719329','3500047241','7175501106','7192199762','1113216870','4149708880','7161100546','3890000640','4149752317','3320009045','5200005156',
		'1450001102','81833601371','7047020558','3007790000120','1500004766','66652294138','66652294136','7116930536','4149720470','2052596957','1004263653622','1020023111','7299235560','2400025605','4149721615','7349151100','69511940400','1530043053','2130014911','1020010513',
		'31722073130','4263400113','4100000920','7349158000','7349152000','4149709518','65398172079','1007080003719','4149766515','5000057989','4149709631','7294075502','81998101208','88491237747','70559901192','88491234944','85210933178','7629501000','4149705183','2550077459',
		'7057504081','1300000036','3732311385','3500099667','31722000490','31722053812','7114010300','7020052204','1007080018043','3320094209','4149713744','77958750417','4149787572','1005783642218','88587542000','1007111702121','5210010779','7629501140','1500018679','64220501896',
		'84747300481','1007080006168','1020022711','2840070007','4149728053','4149719454','4149719404','1121000622','4149708853','7778202988','1629144113','4980000331','4149744939','4116704330','8500003109','5170020622','88173510262','7274500238','4790031316','7321400611',
		'85959400637','81124902032','3800024529','7064002145','3338360351','1629144156','5000057459','7981300011','3077205942','9007080003119','3400003166','81829001678','5010010005','4149709977','36382427864','7096900129','3549389463','6661361056','2400025610','4149703362',
		'3077209401','8831397181','85002007800','85564300605','61124738215','84886004796','1300000125','1800013428','4149700516','4149743678','4120801060','1740022325','1530020067','608002213642','2370005814','2370005057','4110059786','9616286275','57237001030','3810017268',
		'4300009153','8266650050','7170026061','31722051901','23700053312','85509900760','3810018933','8500003171','1650057683','4149703324','7960691025','5150000684','4149708769','76211128952','2500013265','13668000805','31722053960','4149709767','85959400653','3620043151',
		'1064220501551','3160400018','89245300108','1450002757','5100026922','7630175393','4149709250','1600020711','1007080003720','2529300535','80276316869','4148304387','7960607091','74447300028','4149740745','4149719452','76687899943','4149765004','4149734154','85778600813',
		'3700075198','1356200059','85282300610','4593771717','1500007696','7020053014','2515500072','4149719403','8312001011','7535511245','4149711137','3160403041','3077207327','4149743570','3600055228','1007020053423','7287888171','2260008561','74941108125','85210933175',
		'5210005140','4173608006','1600027486','81038702215','7161101952','70405170203','2055911308','3150600010','86287100032','2400000839','4149709966','86407500011','2310013267','9541100062','2840072061','7261373964','2100012380','1700020952','3980008217','88491243494',
		'5210000233','81829001193','4149727812','5100027636','7630175379','1600015235','30087136502','1600049036','1005783642385','5783602266','7170057561','3129014904','4180034900','4920005175','7116925237','7433800331','78302505006','2370005578','4149768030','7766133595',
		'1500000903','81957301647','4149712178','5210004991','1007080004170','2310013775','4263477101','3129014906','4149701054','84667500812','3800020151','4116705075','8500001773','3800015602','88491237811','7684067446','4149766882','1007603335182','4149725146','1005783642828',
		'1001116214279','89353600272','1780019306','3700096894','36382499527','9998807179','84667500391','7119070322','7067271150','4610035568','3450044700','3120001710','85003170037','3160404373','1005783616026','2410011537','4149767784','4139000090','85509900756','85441700201',
		'1700006817','1002480048613','2113100999','4149718286','4149793738','3680014617','7056098255','72745868290','7246597005','4149713742','1629144155','4411540304','4149754298','2780006643','3077208730','2310012237','5210005309','7116924980','4149767782','73052150762',
		'7456241073','2400025436','810065932112','1001123310208','3010044372','7175502574','4149767595','7023010712','7175501066','4411540301','1101740819','3160403284','85314900810','1650059639','4110058930','2791701471','2310013776','4175702327','81829001802','4149708402',
		'8500003218','82946200117','5210003468','4110059419','3680037500','8312001273','3700075186','7005702211','4149709976','7046206212','5000021519','85004220351','4149709981','4069763026','7484718104','4149703527','84667500393','4149756808','60559200752','1600020068',
		'3800022468','73114933836','7161100313','1600018794','7349153000','84667501070','70105','4149740764','84667501350','2420004449','85355200301','82429513715','7294075503','3680019220','2070000500','2400005044','84667501349','7648900473','61124737317','8500003193',
		'2389616670','7007467275','7192145011','4149708344','72243020049','7126229495','4149744940','5260328369','3450015181','60491300102','2130016869','85000968210','5081011406047','30087510107','1600018467','61314064511','8086806019','4149752316','4840000010','4000058614',
		'7110021356','7080004908','7585600072','3800013918','7192189128','1037432004','31722002401','7287881350','3000057543','4149703309','87489600079','794878212558','7274500234','7726009093','7940059443','2310011034','60505258009','4010000771','4149700422','1300000143',
		'1480058223','77958750233','3077102630','1600020775','7064002138','3450015116','84886004333','8705273043','3700087339','7007462082','4149760373','7170021521','7274500224','4173726907','4149708377','4149703539','701604096191','5150078353','7418247350','4980016018',
		'7535511234','4240018899','1005783628017','3077208968','7630184000','5210003823','4149708443','7940058852','1085002719289','89711900101','3500099671','4950825166','3547290220','4149744938','7056098184','7274581441','7146402285','4149708879','3077210051','4149725768',
		'3760083157','81833601017','3700098329','7283067110','85441700219','86000102271','89711900102','2791727173','85562000704','73178912345','7535511244','88133400249','4149712474','5100028730','4149756812','4149740763','5000044449','4956863016','3210020027','5000050362',
		'74859820618','7248600152','4149753056','13668056730','3160404375','3680015124','1085002719284','4460032117','2836420261','83779300399','7648941133','4142005797','4149753044','4149767590','4157014465','86589100014','7020052208','8273464224','501072482983','74447347046',
		'4427602320','87489600580','4650002435','65922378301','7572044706','762935022027','4149708748','7246590851','2068853001','3500056750','3810018931','84667500569','4149711544','1800000206','2370005055','1600017945','4149703161','1600018658','2200028885','3680015123',
		'64963200108','5150024485','7047019111','7750710101','1069511929942','4149728023','4149715370','1530020066','1085001022246','7484700022','4149768365','4149719000','85287000417'
    ]

additional_excluded_columns: list = ['BusinessUnit','ReceiptSfx','UnitOfWeight','VarWghtInd','ItemSize','SourceFile_ATG','ATG_Ref','ATG_ItemID',
                                   'RcptFacility','Merchndsr','Status','ExcptnDate','OKDate','AdjCde1','AdjQty1','AdjCde2','AdjQty2','ListCost','FreeGds',
                                   'WhseDisc','UpDnAmt','UpDnInd','BB','LastCost','LastCost_Orig','CashDisc','FrghtAllwInd','VndrUpDn','VndrUpDnInd',
                                   'PPayAdd','PrePayAddExInd','FrghtBill','Bckhl','VarWght','InvQty','InvListCost','InvOI','InvFreeGds','InvWhseDisc',
                                   'InvUpDnInd','InvBB','InvLastCost','InvLastCost_Orig','InvCashDisc','InvFrghtAllw','InvFrghtAllwExInd','InvVndrUpDn',
                                   'InvPPayAdd','InvFrghtBill','InvBckhl','InvWght','InvVarWght','AdjQty','AdjListCost','AdjOI','AdjFreeGds','AdjWhseDisc',
                                   'AdjUpDnAmt','AdjBB','AdjLastCost','AdjLastCost_Orig','AdjFrghtAllw','DS_Checked','DE_Checked','ClientFamilyID_ATG',
                                   'AdjVndrUpDn','AdjPPayAdd','AdjFrghtBill','AdjBckhl','AdjWght','AdjVarWght','AdjUpDnInd','AdjVndrUpDnInd','Toggle',
                                   'LastRcvCorrDate','Comment','FreeCs','Trans','TransQty','Trans2RcvNbr','Trans2RcvSfx','Trans2Qty','TransFromRcvNbr',
                                   'TransFromRcvSfx','TransFromQty','ItmFrtAllwExInd','AdjPPayAddExInd','FrghtBillExInd','InvFrghtBillExInd','AdjFrghtBillExInd',
                                   'BckhlExInd','InvBckhlExInd','AdjBckhlExInd','DealFlg','DealStatus','InvQualifyAmt','InvFreeCs',
                                   'AdjQualifyAmt','AdjFreeCs','CsUnitFctr','InvSurchgInd','HiOldAvgCost','ACChgTkn','ExcptnUsrID','OKUsrID','WDCostFlg',
                                   'OICostFlg','FGCostFlg','FACostFlg','PPAddCostFlg','BBCostFlg','UDCostFlg','VndUDCostFlg','FBCostFlg','BHCostFlg',
                                   'FullCostFlg','DSDItmDueVndr','ATG_Ref_Orig','ItemFacility','ATG_PO_Ref','IncvOI_ATG','IncvBB_ATG','FlatAmtOI_ATG',
                                   'FlatAmtBB_ATG','Contact','BalFlag_ATG','OOB_ATG','PdHdrFrtVar_ATG','MiscAdj_ATG','ReceiptNbr','PONbr','ReceiptDate',
                                   'POVendorNbr','ItemDescr','ItemNbr','Dept','UPCNbr','ChangeDate_ATG','EndDate_ATG','NextCost_ATG','PrevCost_ATG',
                                   'CostDiff_ATG','IncDecFlag_ATG','CostType_ATG','AddDate_ATG','SellEffDate_ATG','LastShipDate_ATG','ATG_Cost_Ref',
                                   'Facility','ATG_Hdr_Ref','APVendorNbr','InvUpDnAmt','InvVndrUpDnInd','InvPPayAddExInd','UPCUnit','ListCostSB_ATG',
                                   'RepckRatio','QualifyAmt','ShipCube','BestOI_ATG','OI','PODate','PdIncvOI_ATG','PdUpDn_ATG','PdShortOI_ATG','PdFlatOI_ATG']

for upc in upcs_to_keep[104:]:
    train: DataFrame = get_individual_upc(columns=columns,upc=int(upc))
    # train_subset: DataFrame = train[train['UPCNbr'].isin(upcs_to_keep)]
    train_subset_original: DataFrame = train.copy()#train_subset[(train_subset['UPCNbr']==int(upc))]
    # train_subset: DataFrame = train_subset[(train_subset['UPCNbr']==int(upc))]
    train_subset: DataFrame = train.copy()
    train = train.drop(additional_excluded_columns,axis=1)

    print(f"{len(train_subset):,} total rows of data for UPC {upc}.")
    subject = 'Data Collected'
    body = f"""
            {len(train_subset):,} total rows of data for UPC {upc}.
        """
    # send_email_to_self(subject,body)

    for col in train_subset.columns:
        if train_subset[col].dtype == 'object':
            try:
                train_subset[col] = train_subset[col].astype(float)
            except:
                try:
                    train_subset[col] = train_subset[col].astype(int)
                except Exception as e: pass
    print(f"Initial conversions from strings to numbers")
    empty_col = []
    for col in train_subset.columns:
        if train_subset[col].dtype == 'object':
            if(train_subset[col].nunique()>1000):
                print(col)
                empty_col.append(col)
    train_subset = train_subset.drop(empty_col,axis=1)

    for col in train_subset.columns:
        if train_subset[col].dtype == 'object' and not(col=='PdOI_ATG'):
            try:
                train_subset[col] = train_subset[col].astype(float)
            except:
                try:
                    train_subset[col] = train_subset[col].astype(int)
                except Exception as e: pass

    for col in train_subset.columns:
        if train_subset[col].dtype == 'object':
            train_subset[col] = train_subset[col].fillna('None')
        elif train_subset[col].dtype == 'int64':
            train_subset[col] = train_subset[col].fillna(-1)
        elif train_subset[col].dtype == 'float64':
            train_subset[col] = train_subset[col].fillna(-1)

    for col in train_subset.columns:
        if train_subset[col].dtype == 'object' and not(col=='PdOI_ATG'):
            try:
                train_subset[col] = train_subset[col].astype(float)
            except:
                try:
                    train_subset[col] = train_subset[col].astype(int)
                except Exception as e: pass
    print('Last numerical conversion before creating categories')

    category_values: dict = {}
    category_new_values: dict = {}
    for col in train_subset.columns:
        if train_subset[col].dtype == 'object' and train_subset[col].nunique()<=1000 and not(col=='PdOI_ATG'):
            train_subset[col] = train_subset[col].apply(empty_string_to_null)
            # categorical_columns.append(col)
            category_values[col] = Series(list(train_subset[col].unique())).unique()
    
    max_length = 0
    for key,item in category_values.items():
        if(len(item)>max_length):
            max_length = len(item)
    for key in category_values:
        category_values[key] = concatenate((category_values[key],['None'] * (max_length - len(category_values[key]))))
    lookup_table = DataFrame.from_dict(category_values, orient='index')
    lookup_table.columns = range(max_length)
    for col in lookup_table.columns:
        lookup_table[col] = lookup_table[col].astype(str)
    #lookup_table.to_csv(f'C:/Code/Python/Machine_Learning_AI/Lookup_Table_Batch_{batch_number}.csv')
    lookup_table.to_csv(f"C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Lookup_Table.csv")
    del lookup_table

    subject = 'Lookup Table Created'
    body = f'''
            Starting long category_new_values section.
        '''
    # send_email_to_self(subject,body)

    category_new_values = {}
    count = 0
    for key,item in category_values.items():
        # print(f"Categories Read: {count}\nCategories Left: {len(category_values)-count}")
        count += 1
        for new_value in item:
            try:
                category_new_values[f"{key} : {new_value}"] = list(Series(list(train_subset[str(key)].unique())+list(train_subset[str(key)].unique())).unique()).index(new_value)
            except:
                category_new_values[f"{key} : {new_value}"] = len(list(Series(list(train_subset[str(key)].unique())+list(train_subset[str(key)].unique())).unique()))
                break
    del category_values

    count = 0
    for key,item in category_new_values.items():
        full_value = f"{key} : {item}"
        # print(f"Categories Read: {count}\nCategories Left: {len(category_new_values)-count}\n{full_value}\n")
        count += 1
        col = search(r"[A-Za-z\_\s]{1,}:",full_value).group()
        col = full_value.split(':')[0]
        # print(f"{col[:-1]}")
        col_value = search(":.+:",full_value).group()#search(r":[A-Za-z0-9\_\s\-\/\<\>\=\'\.\+\,\&]{1,}:",full_value).group()
        col_value = full_value.split(':')[1]
        # print(col_value[1:-1])
        new_value = search(r": [0-9]{1,}",full_value).group().replace(' ','').replace(':','')
        # print(new_value)
        # print()
        train_subset[col[:-1]] = train_subset[col[:-1]].replace({col_value[1:-1]: new_value})
    del category_new_values

    subject = 'New categories completed'
    body = f"""
            About to prepare for machine learning testing.
        """
    # send_email_to_self(subject,body)

    for col in train_subset.columns:
        if train_subset[col].dtype == 'object':# and not(col=='checked'):
            try:
                train_subset[col] = train_subset[col].astype(float)
            except Exception as e:pass
                # train_subset = DataFrame(train_subset.drop(col,axis=1))
                # test = DataFrame(test.drop(col,axis=1))
                # print(f"{col}: {str(e)}")

    datetime_columns = []
    for col in train_subset.columns:
        if train_subset[col].dtype == 'datetime64[ns]':
            datetime_columns.append(col)
    for col in datetime_columns:
        train_subset[col] = to_numeric(train_subset[col])
    datetime_columns = []
    del datetime_columns

    random_state = 2974306530
    model_scores = DataFrame(None,columns=['Mean_Squared_Error','Root_Mean_Squared_Error','Mean_Absolute_Error','Median_Absolute_Error','Max_Error','R2'])
    options.display.float_format = '{:.10f}'.format
    additional_excluded_columns_old = [
        'IsClaimFlg',
        'Checked',
        'ATG_Ref',
        'ATG_LI_Ref',
        'BatchNbr',
        'CATEGORY_ATG',
        'ClaimType',
        'AP_VndNbr',
        'APVndrName',
        'OOB_ATG',
        'ItemNbr',
        'UPCNbr',
        'UPCUnit',
        'ItemDescription',
        'ItemShipPack',
        'PoNbr',
        'PODate',
        'ReceivingDate',
        'TurnRatio_ATG',
        'DealNbr',
        'OrdStartDate_ATG',
        'OrdEndDate_ATG',
        'DateStartArrival_ATG',
        'DateEndArrival_ATG',
        'DLAmtOI',
        'DLAmtBB'
    ]

    #additional_excluded_columns.extend(empty_col)
    unique_excluded_columns = []
    for col in additional_excluded_columns:
        if col in unique_excluded_columns or not(col in list(train_subset.columns)):
            pass
        else:
            unique_excluded_columns.append(col)
    additional_excluded_columns = unique_excluded_columns.copy()
    
    del unique_excluded_columns
    excluded_features = train_subset[additional_excluded_columns]

    # features = train_subset[train_subset['UPCNbr']==int(upc)].drop(['PdOI_ATG','checked']+additional_excluded_columns,axis=1)
    # target = train_subset[train_subset['UPCNbr']==int(upc)]['PdOI_ATG']
    # print(train_subset.info(max_cols=1000))
    # sleep(100000)
    features = train_subset.drop(['PdOI_ATG']+additional_excluded_columns,axis=1)
    # print(features.info(max_cols=1000))
    # sleep(2000000)
    target = train_subset['PdOI_ATG']

    features_train_subset,features_test,target_train_subset,target_test = train_test_split(features,target,random_state=random_state,test_size=0.5,shuffle=True)

    scaler = StandardScaler()
    scaler.fit(features_train_subset)
    features_train_subset_scaled = DataFrame(scaler.transform(features_train_subset),columns=features.columns)
    features_test_scaled = DataFrame(scaler.transform(features_test),columns=features.columns)
    target_test = target_test.reset_index(drop=True)

    try:
        dc = DummyRegressor(strategy='median')
        dc,dc_train_predictions,dc_predictions,model_scores,dc_model_importances = run_and_analyze_model(dc,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'Dummy')
        print(f"Dummy model predictions complete.")
        subject = 'Dummy Model Predictions Complete'
        body = f""
        # send_email_to_self(subject,body)

        lr = LinearRegression()
        lr,lr_train_predictions,lr_predictions,model_scores,lr_model_importances = run_and_analyze_model(lr,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'Linear')
        print(f"Linear model predictions complete.")
        subject = 'Linear Model Predictions Complete'
        # send_email_to_self(subject,body)

        dt = DecisionTreeRegressor(criterion='absolute_error',random_state=random_state,max_depth=4,max_features=None)
        dt,dt_train_predictions,dt_predictions,model_scores,dt_model_importances = run_and_analyze_model(dt,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'DecisionTree')
        print(f"Decision Tree model predictions complete.")
        subject = 'Decision Tree Model Predictions Complete'
        # send_email_to_self(subject,body)

        rf = RandomForestRegressor(criterion='absolute_error',random_state=random_state,verbose=0,warm_start=True,max_depth=4,n_estimators=100)
        rf,rf_train_predictions,rf_predictions,model_scores,rf_model_importances = run_and_analyze_model(rf,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'RandomForest')
        print(f"Random Forest model predictions complete.")
        subject = 'Random Forest Model Predictions Complete'
        # send_email_to_self(subject,body)

        xgb = XGBRegressor(max_depth=6,random_state=random_state)
        xgb,xgb_train_predictions,xgb_predictions,model_scores,xgb_model_importances = run_and_analyze_model(xgb,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'XGBoost')
        print(f"XGB model predictions complete.")
        subject = 'XGBoost Model Predictions Complete'
        # send_email_to_self(subject,body)

        lgbm = LGBMRegressor(n_estimators=1000,learning_rate=0.01,random_state=random_state,verbose=-1)
        lgbm,lgbm_train_predictions,lgbm_predictions,model_scores,lgbm_model_importances = run_and_analyze_model(lgbm,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'LGBM')
        lgbm_model_importances['LGBM'] = (lgbm_model_importances['LGBM'])/(lgbm_model_importances['LGBM'].sum())
        print(f"LGBM model predictions complete.")
        subject = 'LGBM Model Predictions Complete'
        # send_email_to_self(subject,body)

        gb = GradientBoostingRegressor(loss='absolute_error',warm_start=True,n_estimators=10000,learning_rate=0.0025,random_state=random_state,verbose=0,n_iter_no_change=3)
        gb,gb_train_predictions,gb_predictions,model_scores,gb_model_importances = run_and_analyze_model(gb,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'GradientBoost')
        print(f"Gradient Boost model predictions complete.")
        subject = 'Gradient Boost Model Predictions Complete'
        # send_email_to_self(subject,body)

        knn = KNeighborsRegressor(n_neighbors=3)
        knn,knn_train_predictions,knn_predictions,model_scores,knn_model_importances = run_and_analyze_model(knn,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'KNeighbors')
        print(f"KNeighbors model predictions complete.")
        subject = 'KNearest Neighbors Predictions Complete'
        # send_email_to_self(subject,body)

        cb = CatBoostRegressor(verbose=0,iterations=10000,learning_rate=0.0075,random_state=random_state,early_stopping_rounds=3)
        cb,cb_train_predictions,cb_predictions,model_scores,cb_model_importances = run_and_analyze_model(cb,features_train_subset_scaled,target_train_subset,features_test_scaled,target_test,model_scores,'CatBoost')
        cb_model_importances['CatBoost'] = (cb_model_importances['CatBoost'])/(cb_model_importances['CatBoost'].sum())
        print(f"Cat Boost model predictions complete.")
        subject = 'Cat Boost Model Predictions Complete'
        # send_email_to_self(subject,body)
    except:
        dc_train_predictions=Series(target_train_subset,name='Dummy')
        lr_train_predictions=Series(target_train_subset,name='Linear')
        dt_train_predictions=Series(target_train_subset,name='DecisionTree')
        rf_train_predictions=Series(target_train_subset,name='RandomForest')
        xgb_train_predictions=Series(target_train_subset,name='XGBoost')
        lgbm_train_predictions=Series(target_train_subset,name='LGBM')
        gb_train_predictions=Series(target_train_subset,name='GradientBoost')
        knn_train_predictions=Series(target_train_subset,name='KNeighbors')
        cb_train_predictions=Series(target_train_subset,name='CatBoost')
        dc_predictions=Series(target_test,name='Dummy')
        lr_predictions=Series(target_test,name='Linear')
        dt_predictions=Series(target_test,name='DecisionTree')
        rf_predictions=Series(target_test,name='RandomForest')
        xgb_predictions=Series(target_test,name='XGBoost')
        lgbm_predictions=Series(target_test,name='LGBM')
        gb_predictions=Series(target_test,name='GradientBoost')
        knn_predictions=Series(target_test,name='KNeighbors')
        cb_predictions=Series(target_test,name='CatBoost')
        model_scores.loc['Dummy'] = get_scores(target_test,dc_predictions)
        model_scores.loc['Linear'] = get_scores(target_test,lr_predictions)
        model_scores.loc['DecisionTree'] = get_scores(target_test,dt_predictions)
        model_scores.loc['RandomForest'] = get_scores(target_test,rf_predictions)
        model_scores.loc['XGBoost'] = get_scores(target_test,xgb_predictions)
        model_scores.loc['LGBM'] = get_scores(target_test,lgbm_predictions)
        model_scores.loc['GradientBoost'] = get_scores(target_test,gb_predictions)
        model_scores.loc['KNeighbors'] = get_scores(target_test,knn_predictions)
        model_scores.loc['CatBoost'] = get_scores(target_test,cb_predictions)
        dc_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['Dummy']).T
        lr_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['Linear']).T
        dt_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['DecisionTree']).T
        rf_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['RandomForest']).T
        xgb_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['XGBoost']).T
        lgbm_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['LGBM']).T
        gb_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['GradientBoost']).T
        knn_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['KNeighbors']).T
        cb_model_importances = DataFrame([[1/len(features_train_subset.columns) for _ in range(len(features_train_subset.columns))]],columns=list(features_train_subset.columns),index=['CatBoost']).T

    model_scores.to_csv(f"C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Model_Analysis/Model_Scores/Model_Scores_{upc}.csv")

    feature_importances = concat([dc_model_importances,lr_model_importances,dt_model_importances,rf_model_importances,
                                xgb_model_importances,gb_model_importances,lgbm_model_importances,
                                cb_model_importances,knn_model_importances],axis=1)
    feature_importances['SUM'] = feature_importances['Dummy']+feature_importances['Linear']+feature_importances['DecisionTree']+feature_importances['RandomForest']+feature_importances['XGBoost']+feature_importances['GradientBoost']+feature_importances['LGBM']+feature_importances['CatBoost']+feature_importances['KNeighbors']

    feature_importances.to_csv(f'C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Model_Analysis/Feature_Importances/Feature_Importances_{upc}.csv')

    # scatter_plot_models(target_test,dc_predictions,'Dummy')
    # scatter_plot_models(target_test,lr_predictions,'Linear')
    # scatter_plot_models(target_test,dt_predictions,'DecisionTree')
    # scatter_plot_models(target_test,rf_predictions,'RandomForest')
    # scatter_plot_models(target_test,xgb_predictions,'XGBoost')
    # scatter_plot_models(target_test,gb_predictions,'GradientBoost')
    # scatter_plot_models(target_test,knn_predictions,'KNeighbors')
    # scatter_plot_models(target_test,cb_predictions,'CatBoost')
    # print('Scatter plots plotted')
    train_subset_original['PdOI_ATG'] = train_subset_original['PdOI_ATG'].astype(float)
    features_train_unscaled,features_test_unscaled = train_test_split(train_subset_original,random_state=random_state,test_size=0.5,shuffle=True)
    df_with_predictions_test = concat([DataFrame(features_test_unscaled).reset_index(drop=True),lr_predictions,dt_predictions,rf_predictions,xgb_predictions,gb_predictions,lgbm_predictions,cb_predictions,knn_predictions],axis=1)
    df_with_predictions_train = concat([DataFrame(features_train_unscaled).reset_index(drop=True),
                                        lr_train_predictions,dt_train_predictions,
                                        rf_train_predictions,xgb_train_predictions,gb_train_predictions,
                                        lgbm_train_predictions,cb_train_predictions,knn_train_predictions],axis=1)
    df_with_predictions = concat([df_with_predictions_test,df_with_predictions_train],ignore_index=True,axis=0)
    options.display.float_format = '{:.4f}'.format
    df_with_predictions.sort_values(by='UPCNbr').to_csv(f'Data_With_ML_Predictions_UPC_{upc}.csv',index=False)
    subject = 'Data with Model Predictions Added to CSV'
    # send_email_to_self(subject,body)
    df_with_predictions[(abs(df_with_predictions['DecisionTree']-df_with_predictions['PdOI_ATG'])>(df_with_predictions['PdOI_ATG']/20))&
                        (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['DecisionTree'])>0.001)].to_csv("C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Discrepancies/Worst_Discrepancies.csv",index=False)
    df_with_predictions[
                            (
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['Linear'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['DecisionTree'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['RandomForest'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['XGBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['GradientBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['LGBM'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['CatBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['KNeighbors'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (df_with_predictions['PdOI_ATG']<0.01)
                            )
                        ].to_csv(f"C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Discrepancies/PdOI_Before_AI_0_And_Models_Not_0.csv")
    
    data = df_with_predictions[
                            (
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['Linear'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['DecisionTree'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['RandomForest'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['XGBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['GradientBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['LGBM'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['CatBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['KNeighbors'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (df_with_predictions['PdOI_ATG']<0.01)
                            )
                        ].reset_index(drop=True)
    data.to_csv("C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Discrepancies/All_Model_Predictions_Significantly_Different_From_Original_PdOI_ATG_Where_Pd_OI_Zero.csv",index=False)
    data_2 = df_with_predictions[
                            (
                                (df_with_predictions['Linear']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['DecisionTree']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['RandomForest']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['XGBoost']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['GradientBoost']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['LGBM']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['CatBoost']>(df_with_predictions['PdOI_ATG']))&
                                (df_with_predictions['KNeighbors']>(df_with_predictions['PdOI_ATG']))
                            )
                        ].reset_index(drop=True)
    data_2.to_csv("C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Discrepancies/All_Model_Predictions_Bigger_Than_Original_PdOI_ATG.csv",index=False)
    data_3 = df_with_predictions[
                            (
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['Linear'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['DecisionTree'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['RandomForest'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['XGBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['GradientBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['LGBM'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['CatBoost'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (abs(df_with_predictions['PdOI_ATG']-df_with_predictions['KNeighbors'])>=(df_with_predictions['PdOI_ATG']/5))&
                                (df_with_predictions['PdOI_ATG']>0.01)
                            )
                        ].reset_index(drop=True)
    data_3.to_csv("C:/Code/Python/Machine_Learning_AI/DealLI_Exceptions/Discrepancies/All_Model_Predictions_Significantly_Different_From_Original_PdOI_ATG_Where_Pd_OI_NOT_Zero.csv",index=False)
    
    print(f"{len(df_with_predictions):,} rows being exported to SQL.\n")
    export_to_sql(df_with_predictions)
    # export_to_sql(data_2)
    # export_to_sql(data_3)