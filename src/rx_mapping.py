import xml.etree.ElementTree as ET
import pandas as pd

def extract_drugbank_mappings(xml_file_path):
    print("Starting iterative parsing for UMLS/RxNorm mappings...")
    ns = '{http://www.drugbank.ca}' 
    mappings = []
    
    context = ET.iterparse(xml_file_path, events=('end',))
    
    for event, elem in context:
        # Only look at the main drug entries
        if elem.tag == f'{ns}drug':
            db_id = elem.findtext(f'{ns}drugbank-id[@primary="true"]')
            
            if db_id:
                cui = None
                rxnorm = None
                
                # Hunt for the external identifiers node
                ext_ids = elem.find(f'{ns}external-identifiers')
                if ext_ids is not None:
                    for ext_id in ext_ids.findall(f'{ns}external-identifier'):
                        resource = ext_id.findtext(f'{ns}resource')
                        identifier = ext_id.findtext(f'{ns}identifier')
                        
                        if resource == 'UMLS CUI':
                            cui = identifier
                        elif resource == 'RxCUI':
                            rxnorm = identifier
                
                # Only save if at least one mapping found
                if cui or rxnorm:
                    mappings.append({
                        'drugbank_id': db_id,
                        'cui': cui,
                        'rxnorm': rxnorm
                    })
            
            elem.clear()
            
    df_map = pd.DataFrame(mappings)
    print(f"Extraction complete. Found {len(df_map)} mapped drugs.")
    return df_map

# Run the extraction
df_crosswalk = extract_drugbank_mappings('full database.xml')

# Save so never have to parse the XML for mappings again
df_crosswalk.to_csv('drugbank_cui_crosswalk.csv', index=False)
print("Saved mapping to drugbank_cui_crosswalk.csv")
print(df_crosswalk.head())