def load_data(xml_path: str ='ORTEC01.xml'):
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    # Parse do XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Dicionários para armazenar dados
    shifts = {}                     # ID do turno → informações
    employees = {}                  # ID do funcionário → nome e contrato
    contracts = defaultdict(list)   # ID do contrato → lista de padrões/regras
    cover_requirements = defaultdict(dict)  # Dia da semana → turnos → demanda
    shift_off_requests = []         # Lista de pedidos de folga
    shift_on_requests = []          # Lista de pedidos de turno desejado

    # === 1. Turnos disponíveis ===
    for shift in root.find("ShiftTypes"):
        shift_id = shift.attrib["ID"]
        shifts[shift_id] = {
            "Label": shift.findtext("Label"),
            "Name": shift.findtext("Name"),
            "StartTime": shift.findtext("StartTime"),
            "EndTime": shift.findtext("EndTime")
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

    # === 2. Funcionários ===
    for emp in root.find("Employees"):
        emp_id = emp.attrib["ID"]
        employees[emp_id] = {
            "Name": emp.findtext("Name"),
            "ContractID": emp.findtext("ContractID")
        }

    # === 3. Contratos e padrões ===
    for contract in root.find("Contracts"):
        contract_id = contract.attrib["ID"]
        rules = []
        for match in contract.find("Patterns").findall("Match"):
            rule = {
                "Max": match.findtext("Max/Count"),
                "Weight": match.findtext("Max/Weight"),
                "Label": match.findtext("Max/Label"),
                "Pattern": [el.tag for el in match.findall("Pattern/*")]
            }
            rules.append(rule)
        contracts[contract_id] = rules

    # === 4. Requisitos de cobertura por dia da semana ===
    for day_cover in root.find("CoverRequirements").findall("DayOfWeekCover"):
        day = day_cover.findtext("Day")
        for cover in day_cover.findall("Cover"):
            shift_id = cover.findtext("Shift")
            preferred = int(cover.findtext("Preferred"))
            cover_requirements[day][shift_id] = preferred

    # === 5. Pedidos de folga (ShiftOff) ===
    for req in root.find("ShiftOffRequests").findall("ShiftOff"):
        shift_off_requests.append({
            "EmployeeID": req.findtext("EmployeeID"),
            "ShiftTypeID": req.findtext("ShiftTypeID"),
            "Date": req.findtext("Date"),
            "Weight": int(req.attrib["weight"])
        })

    # === 6. Pedidos de turno desejado (ShiftOn) ===
    for req in root.find("ShiftOnRequests").findall("ShiftOn"):
        shift_on_requests.append({
            "EmployeeID": req.findtext("EmployeeID"),
            "ShiftTypeID": req.findtext("ShiftTypeID"),
            "Date": req.findtext("Date"),
            "Weight": int(req.attrib["weight"])
        })

    # === 7. Datas do período ===
    start_date = root.findtext("StartDate")
    end_date = root.findtext("EndDate")

    # # === Visualização rápida ===
    # print("Turnos:", shifts)
    # print("Funcionários (3 primeiros):", dict(list(employees.items())[:3]))
    # print("Contrato 36 (regras):", contracts["36"][:2])  # exemplo
    # print("Cobertura Segunda-feira:", cover_requirements["Monday"])
    # print("Pedidos de folga:", shift_off_requests[:2])
    # print("Pedidos de turno:", shift_on_requests)

    return {
        "shifts": shifts,
        "employees": employees,
        "contracts": contracts,
        "cover_requirements": cover_requirements,
        "shift_off_requests": shift_off_requests,
        "shift_on_requests": shift_on_requests,
        "start_date": start_date,
        "end_date": end_date
    }

