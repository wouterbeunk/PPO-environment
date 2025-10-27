import os
import pandas as pd

class data_loader:
    def __init__(self, filepath):
        self.file = filepath


    def excel_file_check(self):
        if not os.path.exists(self.file):
            raise FileNotFoundError(self.file,"not found")

    def excel_file_loading(self):
        self.excel_file_check()
        self.filename = os.path.basename(self.file)
        print("Starting to load the Excel File named: ", self.filename)
        extracted_data = {}
        Machine_df = pd.read_excel(self.file, "Stations (free movement)", usecols=["Number", "Machine/workplace/tool (ENG)", "Width", "Height"], engine="openpyxl")
        extracted_data["Stations"] = Machine_df
        Flow_matrix_df = pd.read_excel(self.file, "Liver flow matrix (normalized)", index_col=0, engine="openpyxl")
        extracted_data["Test flow matrix"] = Flow_matrix_df
        #flow_references = list(Flow_matrix_df.columns)
        #print(flow_references)
        #print("the machine sheet: ", "\n", Machine_df.head())
        #print(100*"-")
        #print("flow matrix: ", "\n", Flow_matrix_df.head())
        return extracted_data


if __name__ == "__main__":
    excel_path = "C:\\Users\\beunk\Downloads\\Bij-producten meta data.xlsx"

    tester = data_loader(excel_path)
    sheet = tester.excel_file_loading()

