import os 
from datetime import datetime

# TAG LIST LOADING 
class TagListCheckClass:
    
    def __init__(self, taglist_path, taglist_filename):
            self.taglist_path = taglist_path
            self.taglist_filename = taglist_filename
            self.taglist_file_path = os.path.join(self.taglist_path, self.taglist_filename)

    def load_taglist(self):
        def save_error_report(error_message):
            # Get the current date and time
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

            # Define the report filename
            report_filename = f"error_report_{self.taglist_filename.split('.')[0]}.txt"
            report_path = os.path.join(self.taglist_path, report_filename)

            # Write the error message to the report file
            with open(report_path, 'w') as report_file:
                report_file.write(f"Report generated on: {formatted_datetime}\n\n")
                report_file.write(error_message + "\n\n")

        try:
            with open(self.taglist_file_path, 'r') as taglist_file:
                taglist = [line.strip() for line in taglist_file if line.strip()] 
            return taglist
        except FileNotFoundError as e:
            error_message = f"Tag list file '{self.taglist_file_path}' not found."
            print(error_message)
            save_error_report(error_message)
            raise e
        
        
def run_taglist_load():
    taglist_path = os.getenv("TAGLIST_DIR")  #"C:/Users/cristina.iudica/OneDrive - Bracco Imaging SPA/Documenti/Root_MULTIPLE_SERIES_prova_main/Protocol"
    taglist_filename = "tag_numbers_list_to_anonymize.txt"
    tag_list_loader = TagListCheckClass(taglist_path=taglist_path, taglist_filename=taglist_filename)
    tag_list = tag_list_loader.load_taglist()
    return tag_list
