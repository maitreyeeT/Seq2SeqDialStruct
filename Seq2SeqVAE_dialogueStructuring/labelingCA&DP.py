import pandas as pd
import os


class LabelCaDp():

  def __init__(self):
    dir_path = '//'
    filename = 'Dialog_bankAll_withSynDA3.csv'
    self.readfile = pd.read_csv(os.path.join(dir_path,
                                    filename),
                       sep = '\t')

  def clean_data(self):
    dataframe = self.readfile
    dataframe['Comfun1'] = dataframe.Comfun1.str.replace(
      r'(^.*Answer.*$)', 'Answer')
    dataframe['Comfun1'] = dataframe.Comfun1.str.replace(
        r'(^.*Agree.*$)', 'Agreement')

    dataframe['Comfun1'] = dataframe.Comfun1.str.replace(
        r'(^.*Disagree.*$)', 'Disagreement')

    dataframe['Comfun1'] = dataframe.Comfun1.str.replace(
        r'(^.*AutoPos.*$)', 'Autopositive')


    return dataframe
  def dpLabel(self):
####information seeking communicative functions (all types of questions),
# some directives (Request, Instruct, Suggest)
## discourse structuring
### own and partner communication mgmt (completion)
    DP_data = self.clean_data()

####turn management, time management
##own and partner communication mgmt (completion)
    DP_data['Comfun3'] =DP_data['Comfun1']
    DP_data.loc[DP_data.Comfun3.astype(str).str.lower()
                  .str.contains('turn|stalling|'
                                'pausing|autopositive'
                                '|allopositive'
                                '|autonegative'
                                '|allogenative'
                                '|feedback'
                                '|retraction|selfcorrection|')\
                ,'dialog_policy'] = 'CoOccur'
###all information providing functions
    DP_data.loc[DP_data.Comfun3.astype(str).str.lower()
                  .str.contains('inform|agreement|agree|disagree|answer|'
                                'disagreement|correction'
                                '|confirm|disconfirm|apology|'
                                'thanking|goodbye|'
                                'accept|decline|address')\
                ,'dialog_policy'] = 'Progressive'


    DP_data.loc[DP_data.Comfun3.astype(str)
              .str.lower()
              .str.contains('choicequestion'
                            '|propositionalquestion'
                            '|setquestion|checkquestion'
                            '|question|instruct|'
                            'initialgreeting'
                            '|suggest|request|opening'
                            '|interaction|completion'
                            '|selferror|complain|'
                            'promise') \
  , 'dialog_policy'] = 'Binding'

    return DP_data

  def caLabel(self):
    CA_data = self.dpLabel()

    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains('goodbye|Thanking'),
                'dialog_CA'] = 'FPPpost'
    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains('Returngoodbye|'
                                'Closing|Completion'),
                'dialog_CA'] = 'SPPpost'
    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains(pat = 'Initial|'
                                'Opening'
                                ), 'dialog_CA'] = 'FPPpre'
    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains('Turngrab|Turntake|Turnkeep|Turnaccept|'
                                'Stalling|Pausing|Error|Uncertain|Complain|'
                                'Apology|Retraction|Selfcorrection|Misspeaking|'
                                'Selferror|Inform'),
                'dialog_CA'] = 'FPPinsert'
    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains('Acceptthanking|Acceptfeedback|'
                                'Acceptapology|Error|'
                                'Correctmis|Autopositive|'
                                'Allopositive|Allonegative|'
                                'Autonegative|elicitation|'
                                'Correction|Promise|Clarify'),
                'dialog_CA'] = 'SPPinsert'
    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains('Choicequestion|'
                            'Propositionalquestion'
                            '|Propositionquestion'
                            '|Setquestion|Checkquestion'
                            '|Question|Request|Suggest'
                            '|Interactionstructuring'
                            '|Offer|Instruct') \
      , 'dialog_CA'] = 'FPPbase'
    CA_data.loc[CA_data.Comfun1.astype(str)
    .str.contains('Answer|answer|Acceptrequest|Acceptsuggest|'
                  'Decline|Addressrequest|Addresssuggest|Acceptoffer|'
                  'Rejectoffer|Uncertain|'
                  'Confirm|Disconfirm|'
                  'Agreement|agreement|Diagreement'
                  ), 'dialog_CA'] = 'SPPbase'
    CA_data.loc[CA_data.Comfun1.astype(str)
                  .str.contains('Return|'
                                'Selfintroduction'
                             ), 'dialog_CA'] = 'SPPpre'

    return CA_data


if __name__ == "__main__":
  label = LabelCaDp().caLabel()
  label.to_csv('withDP.csv',sep='\t')
  import matplotlib.pyplot as plt
  label['dialog_CA']\
    .value_counts().plot.bar()
  plt.show()
