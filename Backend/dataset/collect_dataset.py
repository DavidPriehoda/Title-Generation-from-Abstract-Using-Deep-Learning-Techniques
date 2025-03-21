import time
import csv
import urllib
import urllib.request
import xml.etree.ElementTree as ET


categories = ['cs.SI','cs.CV','cs.DS', 'stat.ML', 'math-ph', 'q-bio.SC', 'nlin.CD', 'cond-mat.soft', 'cs.DC', 'cond-mat.dis-nn', 'cs.LO', 'math.AP', 'nucl-ex', 'eess.SY', 'q-bio.PE', 'math.RA', 'cs.SY', 'math.CA', 'cs.IT', 'astro-ph', 'math.CV', 'cs.NA', 'physics.ins-det', 'cmp-lg', 'cs.AR', 'physics.pop-ph', 'astro-ph.GA', 'math.SG', 'cond-mat.stat-mech', 'math.MG', 'physics.flu-dyn', 'math.LO', 'physics.gen-ph', 'cs.MM', 'math.NT', 'math.RT', 'q-bio.MN', 'eess.IV', 'cs.SD', 'physics.med-ph', 'chao-dyn', 'hep-lat', 'physics.ed-ph', 'cond-mat.supr-con', 'cs.CR', 'nlin.CG', 'math.OA', 'math.PR', 'stat.ME', 'q-bio.GN','cs.CY', 'cs.DL', 'cs.CC', 'cond-mat.quant-gas', 'math.CT', 'gr-qc', 'cs.NI', 'stat.CO', 'math.OC', 'cs.DB', 'cs.AI', 'cs.SC', 'cs.HC', 'math.FA', 'math.DS', 'cs.NE', 'math.NA', 'math.AC', 'nlin.PS', 'math.CO', 'math.KT', 'physics.app-ph', 'cs.CL', 'math.QA', 'cs.MS', 'math.GT', 'dg-ga', 'hep-ph', 'math.SP', 'cs.SE', 'q-bio.NC', 'econ.EM', 'physics.space-ph', 'math.DG', 'solv-int', 'eess.SP', 'nlin.SI', 'cond-mat.other', 'math.AT', 'cs.GT', 'astro-ph.IM', 'cs.LG', 'cs.DM', 'math.GN', 'physics.class-ph', 'cs.CE', 'math.HO', 'math.GM', 'cs.ET', 'math.AG']

total_cnt = 0
seen = set()
output = 'arxiv_data.csv'

with open(output, mode='w', newline='') as csv_file:
    fieldnames = ['Title', 'Abstract', 'Category']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for category in categories:
        for i in range(1, 6):
            try:
                url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&date-from_date=2000&date-to_date=2023&start={500*i}&max_results=500'
                data = urllib.request.urlopen(url)
                data = data.read().decode('utf-8')

                ## Parse the XML data to get the title and abstract
                cnt = 0
                root = ET.fromstring(data)
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text

                    # Check if we have already seen this entry
                    if title in seen:
                        continue
                    else:
                        seen.add(title)

                    # Write the data to the CSV file
                    writer.writerow({'Title': title, 'Abstract': abstract, 'Category': category})
                    
                    cnt += 1
                    total_cnt += 1

                print(f'Finished page {i} for category {category}, {cnt} entries found')
                time.sleep(3) # Arxiv API Terms of Use: 'make no more than one request every three seconds, and limit requests to a single connection at a time'
            except Exception as e:
                print(f'Error: {e}')
                continue
print(f'{total_cnt} entries written to {output}')
