import re
log_data = "W1220 17:58:45.973888    21 node_scenario.cpp:289] <INIT> scenario: sync switch from [ hipilot ] to [ shadow ] succeed, using 123947 us, abort_exec: 118950 us, abort_ds: 0 us, resume_ds: 999 us, resume_exec: 3998 us, timer: 0 us"



pattern = r"from \[([^]]+)\] to \[([^]]+)\]"
match = re.search(pattern, log_data)

if match:
    from_value = match.group(1)
    to_value = match.group(2)
    print("From:", from_value)
    print("To:", to_value)
    
    
pattern = r"using (\d+) us"
matches = re.findall(pattern, log_data)

for match in matches:
    print("time cost:", match)