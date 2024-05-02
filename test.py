import unittest
import json
from app import app, valider_donnees

features = ['FLAG_OWN_REALTY',
 'AMT_INCOME_TOTAL',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'REGION_POPULATION_RELATIVE',
 'DAYS_BIRTH',
 'OWN_CAR_AGE',
 'FLAG_MOBIL',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_PHONE',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'COMMONAREA_MEDI',
 'ELEVATORS_MEDI',
 'DAYS_LAST_PHONE_CHANGE',
 'FLAG_DOCUMENT_6',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_8',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_13',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'NAME_CONTRACT_TYPE_Revolving loans',
 'NAME_TYPE_SUITE_Family',
 'NAME_TYPE_SUITE_Other_A',
 'NAME_TYPE_SUITE_Spouse, partner',
 'NAME_FAMILY_STATUS_Separated',
 'OCCUPATION_TYPE_HR staff',
 'OCCUPATION_TYPE_High skill tech staff',
 'OCCUPATION_TYPE_IT staff',
 'OCCUPATION_TYPE_Laborers',
 'ORGANIZATION_TYPE_Business Entity Type 3',
 'ORGANIZATION_TYPE_Construction',
 'ORGANIZATION_TYPE_Culture',
 'ORGANIZATION_TYPE_Electricity',
 'ORGANIZATION_TYPE_Government',
 'ORGANIZATION_TYPE_Industry: type 1',
 'ORGANIZATION_TYPE_Kindergarten',
 'ORGANIZATION_TYPE_Medicine',
 'ORGANIZATION_TYPE_Self-employed',
 'ORGANIZATION_TYPE_Transport: type 1',
 'ORGANIZATION_TYPE_Transport: type 2',
 'ORGANIZATION_TYPE_Transport: type 4',
 'ORGANIZATION_TYPE_University',
 'ORGANIZATION_TYPE_XNA',
 'FONDKAPREMONT_MODE_org spec account',
 'BURO_MONTHS_BALANCE_SIZE_MEAN',
 'BURO_CREDIT_ACTIVE_Active_MEAN',
 'BURO_CREDIT_CURRENCY_nan_MEAN',
 'BURO_CREDIT_TYPE_Car loan_MEAN',
 'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
 'BURO_CREDIT_TYPE_Interbank credit_MEAN',
 'CLOSED_DAYS_CREDIT_ENDDATE_MAX',
 'PREV_AMT_APPLICATION_MIN',
 'PREV_AMT_DOWN_PAYMENT_MAX',
 'PREV_AMT_DOWN_PAYMENT_MEAN',
 'PREV_DAYS_DECISION_MEAN',
 'PREV_CNT_PAYMENT_MEAN',
 'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
 'PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN',
 'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN',
 'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
 'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
 'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN',
 'PREV_CODE_REJECT_REASON_VERIF_MEAN',
 'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',
 'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN',
 'PREV_NAME_GOODS_CATEGORY_Other_MEAN',
 'PREV_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN',
 'PREV_NAME_PORTFOLIO_Cars_MEAN',
 'PREV_NAME_PORTFOLIO_Cash_MEAN',
 'PREV_CHANNEL_TYPE_Regional / Local_MEAN',
 'PREV_CHANNEL_TYPE_Stone_MEAN',
 'PREV_NAME_YIELD_GROUP_XNA_MEAN',
 'PREV_NAME_YIELD_GROUP_high_MEAN',
 'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
 'PREV_PRODUCT_COMBINATION_POS household with interest_MEAN',
 'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
 'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN',
 'PREV_PRODUCT_COMBINATION_POS other with interest_MEAN',
 'APPROVED_AMT_ANNUITY_MAX',
 'APPROVED_AMT_ANNUITY_MEAN',
 'APPROVED_AMT_CREDIT_MAX',
 'APPROVED_APP_CREDIT_PERC_MEAN',
 'APPROVED_APP_CREDIT_PERC_VAR',
 'APPROVED_AMT_DOWN_PAYMENT_MIN',
 'APPROVED_AMT_DOWN_PAYMENT_MAX',
 'APPROVED_HOUR_APPR_PROCESS_START_MEAN',
 'APPROVED_RATE_DOWN_PAYMENT_MIN',
 'APPROVED_RATE_DOWN_PAYMENT_MAX',
 'REFUSED_APP_CREDIT_PERC_MEAN',
 'REFUSED_AMT_DOWN_PAYMENT_MAX',
 'REFUSED_RATE_DOWN_PAYMENT_MIN',
 'REFUSED_RATE_DOWN_PAYMENT_MAX',
 'INSTAL_DPD_SUM',
 'INSTAL_PAYMENT_PERC_MAX',
 'INSTAL_PAYMENT_PERC_SUM',
 'INSTAL_PAYMENT_DIFF_MEAN',
 'CC_CNT_DRAWINGS_POS_CURRENT_MIN']

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict_route(self):
        # Envoyer une requête POST avec des données fictives
        data_correctes = {'data': [[0] * 100],
                          'columns':features}
        response = self.app.post('/predict/', json=data_correctes)

        # Vérifier le code de statut de la réponse
        self.assertEqual(response.status_code, 200)


    def test_valider_donnees(self):
        # Données correctes
        data_correctes = {'data': [[0] * 100],
                          'columns': features}
        
        self.assertIsNone(valider_donnees(data_correctes))  # La fonction ne doit pas retourner d'erreur

        # Données manquantes
        data_manquantes = {'columns': features}
        with self.assertRaises(ValueError):
            valider_donnees(data_manquantes)
            
        # Colonnes manquantes
        col_manquantes = {'data': [[0] * 100]}
        with self.assertRaises(ValueError):
            valider_donnees(col_manquantes)

        # Données incorrectes (longueur incorrecte)
        data_incorrectes_longueur = {'data': [[0] * 99]}
        with self.assertRaises(ValueError):
            valider_donnees(data_incorrectes_longueur)

if __name__ == '__main__':
    unittest.main()
