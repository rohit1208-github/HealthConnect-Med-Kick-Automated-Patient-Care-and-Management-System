import numpy as np

# Setting up doctor and patient information with random attributes
doc_list = [{"doc_id": f"D{i}", "open_days": np.random.choice(range(30), 10, replace=False), "max_patients": np.random.randint(1, 4)} for i in range(1, 21)]
patient_list = [{"pat_id": f"P{i}", "appointment_needs": np.random.choice(range(30), np.random.randint(1, 4), replace=False), "severity": np.random.choice(["High", "Medium", "Low"])} for i in range(1, 101)]

# Mapping severity to a numerical priority
severity_map = {"High": 3, "Medium": 2, "Low": 1}

# Introducing noise for privacy
def noise_for_privacy(dataset, delta=1.0, epsilon=0.1):
    laplace_noise = np.random.laplace(0, delta / epsilon, size=dataset.shape)
    return dataset + laplace_noise

# Aggregating patient updates with privacy considerations
def aggregate_with_privacy(updates, version, delta=1.0, epsilon=0.1):
    updates_sum = np.zeros((30,))
    for update in updates:
        weight = severity_map.get(update.get("severity", "Low"), 1)
        updates_sum[update["appointment_needs"]] += weight * version
    protected_updates = noise_for_privacy(updates_sum, delta, epsilon)
    return np.argsort(protected_updates)[-10:]

# Random selection of a subset of clients
def select_clients(clients, fraction=0.1):
    chosen = np.random.choice(len(clients), size=int(len(clients) * fraction), replace=False)
    return [clients[i] for i in chosen]

# Evaluating the effectiveness of the scheduling
def eval_schedule(schedule, doc_list, patient_list):
    successful_matches = sum(1 for patient in patient_list for doctor in schedule.values() if patient['pat_id'] in doctor and any(day in patient['appointment_needs'] for day in schedule.keys()))
    return successful_matches / len(patient_list)

# Adjusting learning rates based on performance
def learning_rate_adjust(initial_rate, success_rate, cycle):
    if success_rate > 0.8:
        return max(initial_rate / np.log(cycle+1), 0.01)
    else:
        return min(initial_rate * np.log(cycle+1), 1.0)

# Simulating local improvement for doctors
def improve_local(doctor_info, cycles=5):
    for _ in range(cycles):
        if isinstance(doctor_info['max_patients'], (int, float)):
            doctor_info['max_patients'] = doctor_info['max_patients'] * np.random.uniform(0.95, 1.05)
    return doctor_info

# Scheduling with personalized adjustments and robust aggregation
def personalized_scheduling(doc_list, patient_list, cycles=10):
    schedule_plan = {}
    day_to_patient = {day: [] for day in range(30)}
    
    for cycle in range(1, cycles + 1):
        doctor_feedback = [improve_local(doc.copy(), cycles=3) for doc in select_clients(doc_list)]
        patient_feedback = [patient for patient in select_clients(patient_list, fraction=0.2)]
        
        preferred_slots = aggregate_with_privacy(patient_feedback, cycle)
        
        for slot in preferred_slots:
            if slot not in schedule_plan:
                schedule_plan[slot] = []
            eligible_docs = [doc for doc in doctor_feedback if slot in doc['open_days'] and len(schedule_plan[slot]) < doc['max_patients']]
            
            for doc in eligible_docs:
                for patient in patient_feedback:
                    if slot in patient['appointment_needs'] and patient['pat_id'] not in sum(day_to_patient.values(), []):
                        schedule_plan[slot].append(doc['doc_id'])
                        day_to_patient[slot].append(patient['pat_id'])
                        break
                
                if len(schedule_plan[slot]) >= doc['max_patients']:
                    break

    return schedule_plan, day_to_patient

# Running the enhanced scheduling process
final_schedule, final_assignments = personalized_scheduling(doc_list, patient_list, cycles=10)
schedule_efficiency = eval_schedule(final_schedule, doc_list, patient_list)
print(f"Schedule Efficiency: {schedule_efficiency}")

# Displaying the schedule with assignments
for day, doctors in sorted(final_schedule.items()):
    patients_for_day = ", ".join(final_assignments[day])
    print(f"Day {day}: Doctors: {doctors}, Patients: [{patients_for_day}]")
