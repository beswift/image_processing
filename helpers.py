from datetime import datetime
import csv


def parse_opd3_csv(file, delimeter):
    with open(file, newline='') as csvfile:
        # open up a reader object to read in a csv file
        reader = csv.reader(csvfile, delimiter=',', quotechar="'")
        # read object
        read = []
        parsed_fields = []
        # read each row and append to the read object
        for row in reader:
            read.append(row)
        dateLine = str(read[0])
        date = datetime.strptime((dateLine.lstrip("['Date :'").rstrip("']")), "%m/%d/%Y").date().strftime("%m/%d/%Y")
        parsed_fields.append(date)
        pidLine = str(read[1])
        dob = datetime.strptime((pidLine.split(",")[-1].lstrip("'DOB :").rstrip("']")), "%m/%d/%Y").date().strftime(
            ("%m/%d/%Y"))
        parsed_fields.append(dob)
        pid = pidLine.lstrip("['Patient ID :").split(",")[0].rstrip("'")
        parsed_fields.append(pid)
        pNameLine = str(read[2])
        firstname = pNameLine.split(".")[1].rstrip("']")
        parsed_fields.append(firstname)
        lastname = pNameLine.lstrip("['Patient Name:").rsplit(".")[0]
        parsed_fields.append(lastname)
        cDLine = str(read[4])
        capturedate = datetime.strptime((cDLine.split(",")[2].lstrip("'Exam Date :").rstrip("']")),
                                        "%m/%d/%Y %H:%M").strftime(("%m/%d/%Y %H:%M"))
        parsed_fields.append(capturedate)
        return parsed_fields