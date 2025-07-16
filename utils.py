def load_data(xml_path: str ='ORTEC01.xml'):
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    # Parse do XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Dicionários para armazenar dados
    shifts = {}                     # ID do turno -> informações
    employees = {}                  # ID do funcionário -> nome e contrato
    contracts = defaultdict(list)   # ID do contrato -> lista de padrões/regras
    cover_requirements = defaultdict(dict)  # Dia da semana -> turnos -> demanda
    shift_off_requests = []         # Lista de pedidos de folga
    shift_on_requests = []          # Lista de pedidos de turno desejado
    shift_groups = {}
    cover_weights = {}

    # Turnos disponíveis
    for shift in root.find("ShiftTypes"):
        shift_id = shift.attrib["ID"]
        shifts[shift_id] = {
            "Label": shift.findtext("Label"),
            "Name": shift.findtext("Name"),
            "StartTime": shift.findtext("StartTime"),
            "EndTime": shift.findtext("EndTime"),
            "Color": shift.findtext("Color")
        }

    # Adicionar turno especial para folga ("OFF")
    # Se o turno "OFF" não estiver nos dados, adicionamos.
    if "OFF" not in shifts:
        shifts["OFF"] = {
            "Label": "OFF",
            "Name": "Folga",
            "StartTime": None,
            "EndTime": None
        }

    # ShiftGroups
    for group in root.find("ShiftGroups"):
        group_id = group.attrib["ID"]
        shift_list = [s.text for s in group.findall("Shift")]
        shift_groups[group_id] = shift_list

    # Funcionários
    for emp in root.find("Employees"):
        emp_id = emp.attrib["ID"]
        employees[emp_id] = {
            "Name": emp.findtext("Name"),
            "ContractID": emp.findtext("ContractID")
        }

    # Contratos e padrões
    for contract in root.find("Contracts"):
        contract_id = contract.attrib["ID"]
        rules = []
        for match in contract.find("Patterns").findall("Match"):
            rule = {
                "Max": {
                    "Count": match.findtext("Max/Count"),
                    "Weight": match.findtext("Max/Weight"),
                    "WeightFunction": match.find("Max/Weight").attrib.get("function") if match.find("Max/Weight") is not None else None,
                    "Label": match.findtext("Max/Label")
                },
                "Min": {
                    "Count": match.findtext("Min/Count"),
                    "Weight": match.findtext("Min/Weight"),
                    "WeightFunction": match.find("Min/Weight").attrib.get("function") if match.find("Min/Weight") is not None else None,
                    "Label": match.findtext("Min/Label")
                } if match.find("Min") is not None else None,
                "RegionStart": match.findtext("RegionStart"),
                "RegionEnd": match.findtext("RegionEnd"),
                "Pattern": []
            }

            # Pega os elementos de Pattern como dicionários (não só tags)
            for pat in match.findall("Pattern"):
                pat_dict = {}
                for el in pat:
                    pat_dict[el.tag] = el.text
                rule["Pattern"].append(pat_dict)

            rules.append(rule)
        contracts[contract_id] = rules


    # Requisitos de cobertura por dia da semana
    for day_cover in root.find("CoverRequirements").findall("DayOfWeekCover"):
        day = day_cover.findtext("Day")
        for cover in day_cover.findall("Cover"):
            shift_id = cover.findtext("Shift")
            preferred = int(cover.findtext("Preferred"))
            cover_requirements[day][shift_id] = preferred

    # Pedidos de folga (ShiftOff)
    for req in root.find("ShiftOffRequests").findall("ShiftOff"):
        shift_off_requests.append({
            "EmployeeID": req.findtext("EmployeeID"),
            "ShiftTypeID": req.findtext("ShiftTypeID"),
            "Date": req.findtext("Date"),
            "Weight": int(req.attrib["weight"])
        })

    # Pedidos de turno desejado (ShiftOn)
    for req in root.find("ShiftOnRequests").findall("ShiftOn"):
        shift_on_requests.append({
            "EmployeeID": req.findtext("EmployeeID"),
            "ShiftTypeID": req.findtext("ShiftTypeID"),
            "Date": req.findtext("Date"),
            "Weight": int(req.attrib["weight"])
        })

    # CoverWeights
    cover_weights_node = root.find("CoverWeights")
    if cover_weights_node is not None:
        cover_weights["PrefOverStaffing"] = int(cover_weights_node.findtext("PrefOverStaffing", default="0"))
        cover_weights["PrefUnderStaffing"] = int(cover_weights_node.findtext("PrefUnderStaffing", default="0"))

    # Datas do período
    start_date = root.findtext("StartDate")
    end_date = root.findtext("EndDate")

    return {
        "shifts": shifts,
        "shift_groups" : shift_groups,
        "employees": employees,
        "contracts": contracts,
        "cover_requirements": cover_requirements,
        "shift_off_requests": shift_off_requests,
        "shift_on_requests": shift_on_requests,
        "start_date": start_date,
        "end_date": end_date
    }

