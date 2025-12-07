# **🎬 Demo Interaction Script (Deteriorating Patient Case)**

## **Scenario: Complex Cardiac Patient with Deteriorating Course**

### **Clinical Context**
This scenario demonstrates the model's ability to identify patients who are **NOT responding to treatment** - a critical capability for clinical decision support. Unlike cases where risk decreases with intervention, this patient shows a trajectory of increasing risk despite appropriate care, indicating overwhelming disease burden.

### **Setup**
- **Patient**: 78-year-old white male, married, Medicare
- **Condition**: Septic shock with acute coronary syndrome (ACS) and endocarditis
- **Location**: Hospital A Emergency Department → Transfer to Hospital B
- **Time**: 2:00 AM, Saturday (weekend night)
- **Clinical Course**: Admission → 6h update → 24h update (risk increasing)

---

## **🕐 Phase 1: Initial Admission (2:00 AM)**

### **Step 1: Initial Contact**

**Doctor (via Chat UI at https://hospital-a-himas.run.app):**
```
Hello, I have a new ICU admission. Can you help assess the patient?
```

**Agent Response:**
```
I can help assess this patient. To provide an accurate risk prediction, I need 
a few more details about the patient:

Required Information:

1. Age and Gender: What is the patient's age and gender?

2. Admission Type: Was this an EMERGENCY, URGENT, or ELECTIVE admission? 
   (Emergency admissions often have different risk profiles)

3. Time to ICU: How long was the patient in the hospital before ICU admission? 
   (This helps calculate the early ICU score, which is a strong mortality predictor)

4. Admission Source: Where did the patient come from? 
   (e.g., ER, transfer from another hospital, direct admission)

5. Admission Time: What was the exact date and time the patient was admitted 
   to the ICU? (This helps identify weekend/night admissions, which can impact 
   staffing and outcomes)

6. Chief Complaint/Reason for Admission: What are the primary medical issues 
   leading to ICU admission?

Once I have this information, I can proceed with a comprehensive risk assessment.
```

---

### **Step 2: Patient Data Entry (Admission-Time)**

**Doctor:**
```
78-year-old white male, married, Medicare patient. Emergency admission from ER.

Arrived in ICU 2 hours ago (around midnight) with septic shock and chest pain. 
ECG shows ST elevations in anterior leads - likely acute MI. 

He needs advanced cardiac monitoring and possibly emergent cardiac catheterization. 
Also concerned about endocarditis given the sepsis picture - would need infectious 
disease consult.

This is a Saturday night admission, came through our ER.
```

**Agent Response:**
```
[Clinical Decision Agent activating...]
[Loading global federated model from GCS...]
[Fitting preprocessor on training data...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏥 RISK ASSESSMENT COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mortality Risk: 94.2% (HIGH RISK) 🔴
Prediction Mode: ADMISSION-TIME
Confidence: Initial estimate (temporal data unavailable)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key Risk Factors:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 👴 Advanced Age (78 years)
   → Increases baseline mortality risk with reduced physiologic reserve
   → Limited capacity to recover from multi-system insults

2. ⚡ Emergency Admission
   → Suggests acute decompensation rather than planned care
   → Associated with higher acuity and complexity

3. 🚨 Very Early ICU Admission (<6 hours)
   → Patient required ICU within 2 hours of ER arrival
   → Indicates critical illness requiring immediate intensive care
   → Early ICU score: 3/3 (maximum severity)

4. 🦠 Multi-System Disease Burden
   → Septic shock (systemic inflammatory response)
   → Acute MI (cardiac compromise)
   → Suspected endocarditis (infectious + cardiac)
   → Triple-threat presentation significantly elevates risk

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Clinical Recommendation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is a HIGH risk patient requiring:
- Immediate senior intensivist review
- Aggressive resuscitation
- Consideration of specialized interventions

Given the chief complaint of septic shock, acute MI, and possible endocarditis,
the combination of infectious, cardiac, and systemic compromise creates an
extremely challenging clinical scenario.

Next Steps:
I recommend an updated prediction after 6 hours with actual ICU metrics 
(length of stay, transfers) for a more accurate assessment and to monitor 
risk trajectory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now, let's check Hospital A's resources and historical outcomes for similar cases.
```

---

### **Step 3: Resource Assessment**

**Agent Response:**
```
[Treatment Optimization Agent activating...]
[Querying Hospital A resources from BigQuery...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏥 HOSPITAL RESOURCE ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Hospital A CANNOT treat this patient locally.

Missing Resources:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ Advanced cardiac care
❌ Interventional cardiology (for emergent cardiac catheterization)
❌ Infectious disease specialist

Local Survival Estimate: Very Low (<20%)
- Due to critical missing capabilities
- No historical data for similar cases with positive outcomes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 RECOMMENDATION: TRANSFER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Transfer to a facility with:
- Advanced cardiac care
- Interventional cardiology
- Infectious disease specialists

Urgency: HIGH ⚠️

Given the critical nature of the patient's condition and our limited resources,
I will now query peer hospitals for availability and specialized capabilities.
```

---

### **Step 4: Cross-Hospital Consultation**

**Agent Response:**
```
[Case Consultation Agent activating...]
[Privacy Guardian: Validating data anonymization...]
[Contacting Federated Coordinator...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 PEER HOSPITAL CONSULTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Connectivity Issue: Unable to connect to the Federated Coordinator 
to query peer hospitals programmatically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recommendation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Due to the HIGH URGENCY and critical nature of this patient's condition, 
I recommend initiating a MANUAL CALL to Hospital B immediately to 
inquire about their availability for:

✓ Advanced cardiac care
✓ Interventional cardiology  
✓ Infectious disease specialist consultation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ CRITICAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This patient has a 94.2% (HIGH) mortality risk at Hospital A due to our 
inability to provide the necessary specialized care for:
- Septic shock
- Acute MI  
- Possible endocarditis

TRANSFER IS CRITICALLY URGENT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Transfer Initiation

**Doctor:**
```
Initiate the transfer
```

**Agent:**
```
TRANSFER CONFIRMED ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transfer ID: transfer_20251204_234117
Status: CONFIRMED
From: Hospital A → To: Hospital B

Bed Reservation: ICU_BED_1 in MICU
Receiving Team: Notified ✓
Estimated Transport: 45 minutes
Urgency: HIGH (risk score: 94.1%)

Privacy Guarantees:
✓ Patient data anonymized (age bucket: 75-80)
✓ K-anonymity (k≥5)
✓ Differential privacy (ε=0.1)
✓ HIPAA compliant
✓ Audit logged at BOTH hospitals

Please prepare the patient for transport.
```

---

## **🕐 Phase 2: Six-Hour Update (8:00 AM) - Risk INCREASING**

**Doctor:**
```
Update on the 78yo male we transferred. He arrived at Hospital B and went straight 
to the cath lab. They found a 95% LAD occlusion and placed a stent. He's been in 
their ICU for 6 hours now, total. He was transferred once from their MICU to CCU 
after the procedure. Can you give me an updated risk assessment?
```

**Agent Response:**
```
[Clinical Decision Agent activating...]
[Loading cached model and preprocessor... ✅]
[Processing UPDATED prediction with temporal data...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏥 UPDATED RISK ASSESSMENT (6-Hour Follow-Up at Hospital B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mortality Risk: 96.1% (HIGH RISK) 🔴 ⬆️ INCREASED

Risk Trajectory:
┌─────────────────────────────────────────────────────────────────┐
│  Admission (Hospital A):    94.2% (HIGH)                        │
│  6-Hour Update (Hospital B): 96.1% (HIGH) ⬆️ +1.9%              │
└─────────────────────────────────────────────────────────────────┘

Change: +1.9 percentage points (from previous assessment)

Prediction Mode: UPDATED (Temporal Data Available)
Confidence: High confidence (actual ICU metrics used)
```

---

## **🕐 Phase 3: Twenty-Four Hour Update - Risk STILL INCREASING**

**Doctor:**
```
24-hour update. Patient is still at Hospital B. Total ICU time is now 24 hours. 
He's had 2 total transfers (MICU → CCU → back to MICU after stabilization). 
Blood cultures came back positive - confirmed endocarditis. ID team has him on 
targeted antibiotics. Cardiac echo shows improved function post-stenting.
```

**Agent Response:**
```
[Clinical Decision Agent activating...]
[Processing 24-hour temporal update...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏥 UPDATED RISK ASSESSMENT (24-Hour Follow-Up at Hospital B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mortality Risk: 96.7% (HIGH RISK) 🔴 ⬆️ INCREASED

Risk Trajectory (CONCERNING TREND):
┌─────────────────────────────────────────────────────────────────┐
│  Admission (Hospital A):     94.2% (HIGH)        baseline       │
│  6-Hour Update (Hospital B):  96.1% (HIGH)  ⬆️  +1.9%          │
│  24-Hour Update (Hospital B): 96.7% (HIGH)  ⬆️  +0.6%          │
│                                                                 │
│  Total Increase: +2.5 percentage points over 24 hours ⚠️        │
│  Trend: STEADILY WORSENING                                      │
└─────────────────────────────────────────────────────────────────┘

Prediction Mode: UPDATED (24-Hour Temporal Data)
Confidence: VERY HIGH (full clinical trajectory available)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Temporal Data Used (24-Hour Data):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ICU length of stay: 24 hours (1.0 days)
✅ ICU transfers: 2 (MICU → CCU → MICU)
✅ Distinct units: 2

Key Risk Factors (Updated):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 👴 Advanced Age (78 years)
   → Increases baseline mortality risk with reduced physiologic reserve

2. ⚡ Emergency Admission
   → Suggests acute decompensation rather than planned care

3. 🔄 Multiple ICU Transfers (2)
   → Indicates patient instability or complex care needs
   → Each transfer associated with increased mortality risk
   → MICU → CCU → MICU pattern suggests fluctuating clinical course

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Clinical Interpretation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The patient's mortality risk remains EXTREMELY HIGH at 96.7% after 
24 hours in the ICU, indicating SEVERE critical illness.

Mixed Clinical Picture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POSITIVE DEVELOPMENTS:
✅ Cardiac function improved post-stenting
✅ Targeted antibiotics initiated for confirmed endocarditis
✅ Patient survived critical first 24 hours

CONCERNING FACTORS:
⚠️ Risk has INCREASED rather than decreased
⚠️ Multiple ICU transfers indicate ongoing instability
⚠️ Multi-system involvement (cardiac + infectious + sepsis)
⚠️ Age-related limited physiologic reserve

Why the risk hasn't improved:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The model has learned from our federated network (100+ patients across 
3 hospitals) that patients with this presentation pattern:

1. MULTIPLE ICU TRANSFERS are a strong negative prognostic indicator
   - The MICU → CCU → MICU pattern suggests:
     • Initial instability requiring specialized cardiac care
     • Return to general ICU indicates ongoing concerns
     • Not a simple "step down" trajectory

2. CONFIRMED ENDOCARDITIS significantly complicates prognosis
   - 6-week antibiotic course required
   - Risk of valve complications, embolic events
   - Ongoing septic physiology despite antibiotics

3. CUMULATIVE PHYSIOLOGIC STRESS
   - Even with improved echo, the patient's body has endured:
     • Acute MI with coronary occlusion
     • Emergent catheterization and stenting
     • Active endocarditis
     • Septic shock
   - At 78 years, recovery capacity is limited

4. The model identifies that patients who don't show DECREASING risk 
   by 24 hours despite interventions have persistently poor outcomes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 CRITICAL RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is a CRITICAL risk patient. Immediate actions warranted:

1. SENIOR INTENSIVIST REVIEW (Mandatory)
   - Re-evaluate current treatment strategy
   - Assess for occult sources of deterioration
   - Consider additional imaging/diagnostics

2. ADVANCED ORGAN SUPPORT EVALUATION
   - ECMO candidacy assessment
   - CRRT evaluation if renal function compromised
   - Advanced hemodynamic monitoring

3. SUBSPECIALTY CONSULTATIONS
   - Cardiothoracic surgery (endocarditis + valve assessment)
   - Interventional cardiology (for any stent complications)
   - Infectious disease (antibiotic optimization, source control)

4. ⚠️ GOALS OF CARE DISCUSSION
   Given the extremely high mortality risk (96.7%) and worsening 
   trajectory, a comprehensive discussion of goals of care with 
   the family is STRONGLY RECOMMENDED.

   Topics to address:
   - Current prognosis and risk trajectory
   - Treatment options and their limitations
   - Patient's previously expressed wishes
   - Comfort care vs. aggressive intervention
   - Healthcare proxy decisions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Risk Trajectory Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time Point          Risk Score    Change    Interpretation
─────────────────────────────────────────────────────────────────
Admission (0h)      94.2%         --        Initial high risk
6-Hour Update       96.1%         +1.9%     Worsening despite PCI
24-Hour Update      96.7%         +0.6%     Continued worsening

Overall Trend: ⬆️ DETERIORATING (+2.5% over 24 hours)

The lack of improvement despite interventions suggests this patient 
may not survive the current hospitalization. Palliative care 
consultation should be considered.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
---

Here are user prompts that would result in a **POSITIVE response** (mortality risk DECREASING over time):

---

## **Positive Case: User Prompts for Improving Patient**

### **Initial Admission Prompt**

```
Hello, I have a new ICU admission. Can you help assess the patient?
```

```
65-year-old white female, married, private insurance. Emergency admission from ER.

Arrived in ICU 4 hours ago with community-acquired pneumonia and respiratory 
distress. She was hypoxic in the ER but responding well to supplemental oxygen. 
No chest pain, ECG is normal sinus rhythm.

She has well-controlled hypertension and type 2 diabetes. No history of heart 
disease. This is a Monday morning admission.
```

**Expected Result:** Lower initial risk (~60-70%) due to:
- Younger age (65 vs 78)
- Single-organ involvement (respiratory only)
- No cardiac complications
- Weekday daytime admission (better staffing)
- Slightly longer time to ICU (4h vs 2h = less acute)

---

## **Alternative Prompts for Different Positive Scenarios**

### **Scenario B: Post-Surgical Patient Doing Well**

### Initial Assessment

**Doctor:**
```
Hello, I have a new ICU admission. Can you help assess the patient?
```

**Agent:**
```
I can help assess this patient. Please provide:
- Age and gender
- Admission type (emergency, urgent, elective)
- How long ago admitted to ICU
- Primary reason for ICU admission
```

---

**Doctor:**
```
58-year-old Asian male, employed with private insurance. Elective admission for planned CABG surgery. He's now 3 hours post-op in the cardiac surgery ICU. Surgery went smoothly, no complications. He's intubated per protocol but hemodynamically stable on minimal pressors.
```

**Agent:**
```
RISK ASSESSMENT COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mortality Risk: 72.1% (HIGH RISK)
Prediction Mode: ADMISSION-TIME
Early ICU Score: 3 (Very Early - <6 hours post-op)

Key Risk Factors:
• High risk despite elective admission and stable status
• Early ICU admission (<6 hours post-op)
• Age (58 years)

Recommendation: Updated prediction after 6 hours with actual ICU metrics.
```

---

### 6-Hour Update

**Doctor:**
```
6-hour update on the post-CABG patient. He's been extubated successfully, no transfers needed. He's in the same CSICU bed, tolerating oral intake, and his cardiac output is excellent. Chest tubes have minimal drainage.
```

**Agent:**
```
UPDATED RISK ASSESSMENT (6-Hour Follow-Up)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mortality Risk: 72.1% (HIGH RISK) ↔ STABLE

Risk Trajectory:
• Admission: 72.1% (HIGH)
• 6-Hour Update: 72.1% (HIGH)
• Change: 0 percentage points

Temporal Data Used:
✓ ICU length of stay: 6 hours
✓ ICU transfers: 0
✓ Extubated: Yes

Clinical Interpretation: Risk remains HIGH despite positive clinical improvements. 
Continue close monitoring. Senior intensivist review recommended.
```

---

### 24-Hour Update

**Doctor:**
```
24-hour update. Patient still in CSICU, no transfers. He's ambulating with physical therapy, chest tubes removed, and we're planning floor transfer tomorrow. Textbook recovery.
```

**Agent:**
```
UPDATED RISK ASSESSMENT (24-Hour Follow-Up)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mortality Risk: 72.2% (HIGH RISK) ⬆ SLIGHT INCREASE

Risk Trajectory:
• Admission: 72.1%
• 6-Hour: 72.1%
• 24-Hour: 72.2% (+0.1%)

Clinical Interpretation: Despite "textbook recovery," model risk remains HIGH. 
This case valuable for future model refinement.

Recommendation: Continue floor transfer as clinically indicated.
```

---

### **Scenario C: Younger Trauma Patient**

**Admission:**
```
42-year-old Hispanic male, employed. Emergency admission after motor vehicle 
accident. Arrived in trauma ICU 5 hours after ER presentation - he was stable 
enough for full trauma workup first. He has a splenic laceration (grade 2, 
managed non-operatively) and rib fractures. No head injury, GCS 15.
```

**6-Hour Update:**
```
6-hour update on the trauma patient. No transfers, same trauma ICU bed. 
Serial hemoglobin stable, no signs of ongoing bleeding. Pain well controlled. 
He's alert and following commands appropriately.
```

**24-Hour Update:**
```
24-hour update. Still in trauma ICU, no transfers. Repeat CT shows stable 
splenic laceration, no expansion. He's eating, ambulating to chair, and 
we're targeting step-down transfer in 24-48 hours.
```

---

## **Key Factors That Drive DECREASING Risk**

| Factor | High Risk (Increasing) | Low Risk (Decreasing) |
|--------|------------------------|----------------------|
| Age | 78 years | 58-65 years |
| Admission Type | EMERGENCY | ELECTIVE or less acute EMERGENCY |
| Time to ICU | <2 hours (very early) | 4-6 hours (less acute) |
| ICU Transfers | Multiple (2+) | Zero |
| Distinct ICU Units | 2-3 | 1 |
| Admission Time | Weekend night | Weekday daytime |
| Organ Systems | Multi-system (cardiac + sepsis + infectious) | Single system |
| Disease Complexity | MI + Sepsis + Endocarditis | Single diagnosis |


---

# **Federated Coordinator Agent - Standalone Queries**

The Federated Coordinator runs as a standalone A2A service and handles cross-hospital queries with privacy guarantees.

**Endpoint:** `http://localhost:8001` (or deployed Cloud Run URL)
**Agent Card:** `http://localhost:8001/.well-known/agent-card.json`

---

## **Query 1: Hospital Capability Check**

**User Prompt:**
```
Which hospitals in our network have ECMO capability?
```

**Expected Agent Behavior:**
- Queries all hospitals for the specified capability
- Returns list of hospitals with/without the capability
- Includes hospital tier and ICU bed availability
- Provides recommended transfer target (prioritizes tertiary centers)

---

## **Query 2: Specific Capability Query**

**User Prompt:**
```
Does Hospital B have advanced cardiac care and interventional cardiology?
```

**Expected Agent Behavior:**
- Queries Hospital B's specialized_capabilities
- Returns yes/no for each capability
- Includes hospital tier information
- Notes any related capabilities available

---

## **Query 3: Similar Case Consultation (Privacy-Preserved)**

**User Prompt:**
```
Query peer hospitals for outcomes on similar patients. 
Patient profile: age 75-80, emergency admission, early ICU score of 3.
The requesting hospital is hospital_a.
```

**Expected Agent Behavior:**
- Applies k-anonymity (minimum 5 similar patients required)
- Applies differential privacy (ε=0.1 noise added)
- Returns aggregated statistics per hospital:
  - Cases found (noisy count)
  - Survival rate
  - Average ICU LOS
  - Average hospital LOS
- Excludes requesting hospital from results
- Notes privacy guarantees applied

---

## **Query 4: Network Statistics**

**User Prompt:**
```
Give me an overview of our federated network statistics.
```

**Expected Agent Behavior:**
- Returns network-wide aggregated statistics:
  - Total hospitals (3)
  - Hospital names/tiers
  - Total patients across network
  - Overall mortality rate
  - Federated model version
- Per-hospital summaries:
  - Patient count
  - Mortality rate
  - Average age
  - Average ICU LOS
- Data split breakdown (train/validation/test)
- Notes all statistics are privacy-preserved

---

## **Query 5: Transfer Initiation**

**User Prompt:**
```
Initiate a transfer from hospital_a to hospital_b for a patient needing 
advanced cardiac care. Patient fingerprint is abc123, urgency is HIGH.
Transfer reason: ST-elevation MI requiring emergent catheterization.
```

**Expected Agent Behavior:**
- Checks target hospital capacity
- Creates transfer record with:
  - Transfer ID
  - Status (CONFIRMED/FAILED)
  - Bed reservation details
  - Estimated transport time
  - Receiving team notification status
- Returns specialists available at target
- Logs to audit trail at both hospitals

---

## **Query 6: Transfer Status**

**User Prompt:**
```
What is the status of transfer xfer_20251205_143022?
```

**Expected Agent Behavior:**
- Looks up transfer by ID
- Returns current status, source/target hospitals, urgency
- Returns initiated timestamp and transport estimate
- Indicates if transfer ID not found

---

## **Query 7: Patient Data Anonymization**

**User Prompt:**
```
Anonymize patient data for cross-hospital query. 
Subject ID: 12345, Age: 78, Risk score: 0.85
```

**Expected Agent Behavior:**
- Creates SHA-256 patient fingerprint (with daily salt)
- Converts age to 5-year bucket (e.g., "75-80")
- Generalizes risk score to range (e.g., "0.8-0.9")
- Lists identifiers removed
- Confirms k-anonymity compliance

---

# **Known Limitations**

## **Hospital A Agent Cannot:**

| Query Type | Example Prompt | Why It Fails |
|------------|----------------|--------------|
| Staff scheduling | "Is there a cardiologist on call tonight?" | Outside agent capabilities - check hospital's internal on-call schedule |
| Specific equipment inventory | "Do we have any ventilators available?" | Limited ability to check specific equipment; provides general capacity only |
| Patient PII | "What is the patient's name and MRN?" | Privacy constraints - agent cannot access/display PII |
| Non-ICU predictions | "Assess this patient on the general medical floor" | Model trained on ICU data only |

## **Federated Coordinator Cannot:**

| Query Type | Example Prompt | Why It Fails |
|------------|----------------|--------------|
| Individual patient outcomes | "What happened to the 78yo we transferred last week?" | k-anonymity prevents individual lookups |
| Real-time bed updates | "Show me live bed status across all hospitals" | Queries metadata table, not real-time systems |
| Cross-hospital patient matching | "Is patient X at Hospital B?" | Privacy constraints prevent patient identification |

---

# **Quick Reference Tables**

## **Risk Score Interpretation**

| Risk Range | Level | Interpretation |
|------------|-------|----------------|
| 0-50% | LOW-MODERATE | Standard monitoring |
| 50-70% | MODERATE | Enhanced monitoring, consider specialist consult |
| 70-90% | HIGH | Aggressive intervention, senior review |
| >90% | CRITICAL | Goals of care discussion, maximum support |

## **Early ICU Score**

| Hours to ICU | Score | Interpretation |
|--------------|-------|----------------|
| <6 hours | 3 | Very early - critical illness |
| 6-24 hours | 2 | Early - significant illness |
| 24-48 hours | 1 | Delayed - less acute |
| >48 hours | 0 | Late - hospital-acquired deterioration |

## **Risk Trajectory Patterns**

| Pattern | Transfers | Interpretation |
|---------|-----------|----------------|
| Risk DECREASING | 0 | Positive response to treatment |
| Risk STABLE | 0-1 | Stable course, continue monitoring |
| Risk INCREASING | 1+ | Not responding, consider escalation |
| Risk INCREASING | 2+ | Deteriorating, goals of care discussion |

## **Working Prompt Templates**

### **Initial Assessment**
```
[Age]-year-old [gender], [employment/insurance]. [Admission type] admission.
Arrived in ICU [X] hours ago with [chief complaint].
[Additional clinical details].
This is a [day/time] admission.
```

### **Update Assessment**
```
[X]-hour update on the [descriptor] patient. [He/She]'s been in ICU for 
[X] hours total. [Number] transfers - [transfer details if any]. 
[Current clinical status].
```

### **Resource Check**
```
yes check resources
```
or
```
What's our current ICU capacity?
```

### **Federated Query**
```
Query peer hospitals for [capability/outcomes]. 
Patient profile: age [bucket], [admission type], early ICU score [0-3].
Requesting hospital is [hospital_a/b/c].
```

---


## Resource & Capability Queries

### ICU Capacity Check

**Doctor:**
```
What's our current ICU capacity? We might have another admission coming.
```

**Agent:**
```
HOSPITAL A ICU CAPACITY STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Occupancy: 88% (7 of 8 beds occupied)
Status: MODERATE CAPACITY
Available ICU Beds: 1

Limitations:
• No advanced cardiac care unit
• No ECMO capability

We have 1 ICU bed available. Let me know if you need to assess the new admission.
```

---

### ECMO Capability Query

**Doctor:**
```
Which hospitals in our network have ECMO capability?
```

**Agent:**
```
To query peer hospitals for ECMO capability, I need patient context for privacy-preserving query:
- Patient's age at admission
- Admission type (emergency, urgent, elective)
- Early ICU score (0-3)
- Risk score (if available from previous assessment)
```

---

**Doctor:**
```
Query for the 58-year-old Asian male
```

**Agent:**
```
🌐 PEER HOSPITAL CONSULTATION (via A2A)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Capability Query for ECMO (Age 58, Elective, Early ICU Score 3):

Hospital B (Tertiary Care Center):
✓ ECMO capability available
✓ 1 ICU bed available
Similar cases: 16.7% survival rate

Hospital C (Rural Hospital):
✗ ECMO capability not available
Similar cases: 80% survival rate

Privacy Guarantees: K-anonymity (k≥5), Differential privacy (ε=0.1), HIPAA compliant

RECOMMENDATION: If ECMO clinically indicated, Hospital B is the only option.
Would you like me to initiate a transfer request?
```

---

## Scenario 4: Audit Trail Queries

### Privacy-Respecting Audit Query

**Doctor:**
```
Show me the complete audit trail for the 78yo patient we transferred this morning.
```

**Agent:**
```
Due to strict privacy protocols and HIPAA compliance, I cannot retrieve audit logs using patient-specific demographic information like age.

To find the audit trail, please provide:
• Transfer ID (if available)
• User ID of the clinician who initiated the transfer
• Specific timeframe (e.g., "between 2 AM and 3 AM this morning")

This ensures privacy-preserving audit queries.
```

---

### Time-Based Audit Query

**Doctor:**
```
Show me all transfer audit logs from the last 24 hours
```

**Agent:**
```
TRANSFER AUDIT LOGS (Last 24 Hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transfer Initiation Logs (from Hospital A):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Log ID: ext_20251206_040222_33e3849c
Timestamp: 2025-12-06T04:02:22.504880+00:00
Source Hospital: Hospital A
Action: External Query (Transfer Initiation)
Target Hospital: Hospital B
Patient Age Bucket: 75-80 years old
Risk Score: 0.956 (HIGH risk)
Privacy Level: K-anonymity (k=5), Differential Privacy (ε=0.1)
HIPAA Compliant: Yes

Transfer Receipt Logs (for Hospital A):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No incoming transfers in the last 24 hours.
```

---

## Quick Reference: Working Prompts

| Query Type | Example Prompt |
|------------|----------------|
| Initial Assessment | `Hello, I have a new ICU admission. Can you help assess the patient?` |
| Patient Details | `[Age]-year-old [gender], [insurance]. [Admission type] admission. Arrived in ICU [X] hours ago with [chief complaint].` |
| Follow-up Update | `[X]-hour update on the [patient descriptor]. [Current clinical status]. [Number] transfers.` |
| ICU Capacity | `What's our current ICU capacity?` |
| Capability Query | `Which hospitals in our network have [capability]?` |
| Transfer Initiation | `Initiate the transfer` |
| Audit (Time-based) | `Show me all transfer audit logs from the last 24 hours` |
| Audit (Transfer ID) | `Show me the audit trail for transfer ID [transfer_id]` |

---

## Prompts That Won't Work (Privacy Protected)

| Query Type | Example | Why It Fails |
|------------|---------|--------------|
| Patient demographics | `Show audit for the 78yo patient` | Cannot query by age/demographics |
| Individual outcomes | `What happened to the patient we transferred?` | K-anonymity prevents individual lookups |
| Patient PII | `What is the patient's name?` | Privacy constraints |