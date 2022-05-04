import gzip
import sys
from lxml import etree
import pandas as pd

def parse_drugbank():
    input = 'full_database.xml'
    tree = etree.ElementTree(file=input)    #载入数据
    ns = tree.getroot().nsmap
    ns['db'] = ns[None]
    del ns[None]
    root = tree.getroot()
    drugs = root.findall("db:drug", namespaces=ns)
    genes = {}
    for drug in drugs:
        # get its  name
        name = drug.find("db:name", namespaces=ns)
        if name is None:
            continue
        # figure out if approved
        is_approved = False
        groups = drug.find("db:groups", namespaces=ns)
        if groups is not None:
            for group in groups.findall("db:group", namespaces=ns):
                if group.text == "approved":
                    is_approved = True
        if not is_approved:
            continue
        description = drug.find("db:description", namespaces=ns).text
        if description is None or ('cancer' not in description.lower() and 'tumor' not in description.lower()):
            continue
        targets_element = drug.find("db:targets", namespaces=ns)
        if targets_element is not None:
            targets = targets_element.findall("db:target", namespaces=ns)
            for target in targets:
                # not all targets have a position, but for those that do,
                # only take the top-ranked target association
                if not "position" in target.attrib:
                    continue
                known_action = target.find("db:known-action", namespaces=ns).text
                if known_action != "yes":
                    continue
                polypeptide = target.find("db:polypeptide", namespaces=ns)
                if polypeptide is None:
                    continue
                gene_id = polypeptide.find("db:gene-name", namespaces=ns).text
                if gene_id is not None:
                    genes[gene_id] = description
    return genes


if __name__ == '__main__':
    drug_gene = parse_drugbank()
    out ='gene_list.csv'
    df = pd.DataFrame.from_dict(drug_gene, orient="index").reset_index()
    df.columns = ['gene', 'description']
    df.to_csv(out, header=True, index=True, sep=',')
