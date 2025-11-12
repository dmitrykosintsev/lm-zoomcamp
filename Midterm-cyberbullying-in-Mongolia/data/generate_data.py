"""
Cyberbullying Synthetic Dataset Generator
Based on UNDP Mongolia Youth Cyberbullying Survey (March 2025)

Usage:
    python generate_dataset.py

This will create:
    - students_data.csv (student records)
    - parents_data.csv (parent records)
    - combined_data.csv (all records together)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_student_data(n_students):
    """Generate synthetic student data based on survey statistics"""

    students = []

    for i in range(n_students):
        # Age group distribution (32.4% aged 9-12, 67.6% aged 13-17)
        age_group = '9-12' if np.random.random() < 0.324 else '13-17'

        # Gender (53.7% female, 46.3% male)
        gender = 'Female' if np.random.random() < 0.537 else 'Male'

        # Internet usage patterns
        if age_group == '9-12':
            # 66.7% use internet daily
            if np.random.random() < 0.667:
                daily_hours = np.random.randint(1, 5)
            else:
                daily_hours = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        else:
            # 53.6% spend >4 hours daily
            if np.random.random() < 0.536:
                daily_hours = np.random.randint(4, 9)
            else:
                daily_hours = np.random.randint(1, 5)

        # Primary internet activities
        activities = ['Chatting', 'Studying', 'Gaming', 'Social Media', 'Videos']
        primary_activity = np.random.choice(activities)

        # Facebook usage (90% use Facebook despite age restrictions)
        uses_facebook = np.random.random() < 0.90

        # Multiple accounts (33% have multiple accounts)
        has_multiple = np.random.random() < 0.33
        num_accounts = np.random.randint(2, 5) if has_multiple else 1

        # Exposure to inappropriate content
        if age_group == '9-12':
            exposed_bad_language = np.random.random() < 0.536
            learned_bad_words = exposed_bad_language and np.random.random() < 0.536
        else:
            exposed_bad_language = np.random.random() < 0.60
            learned_bad_words = exposed_bad_language and np.random.random() < 0.45

        # Cyberbullying experience
        if age_group == '9-12':
            experienced_cyberbullying = np.random.random() < 0.333  # 33%
        else:
            experienced_cyberbullying = np.random.random() < 0.513  # 51.3%

        # Types of cyberbullying (if experienced)
        cyberbullying_types = []
        if experienced_cyberbullying:
            if np.random.random() < 0.60:
                cyberbullying_types.append('Insults')
            if np.random.random() < 0.45:
                cyberbullying_types.append('Defamation')
            if np.random.random() < 0.25:
                cyberbullying_types.append('Exclusion')
            if np.random.random() < 0.15:
                cyberbullying_types.append('Threats')
            if age_group == '13-17' and np.random.random() < 0.05:
                cyberbullying_types.append('Sexual Harassment')

        # Reporting behavior (50% don't report)
        reported_incident = None
        reported_to = None
        if experienced_cyberbullying:
            reported_incident = np.random.random() < 0.50
            if reported_incident:
                reported_to = np.random.choice(['Family', 'Friend', 'Teacher', 'School Counselor'])

        # Mental health impacts
        mental_health_impact = []
        if experienced_cyberbullying:
            if np.random.random() < 0.318:
                mental_health_impact.append('Insomnia')
            if np.random.random() < 0.295:
                mental_health_impact.append('Academic Decline')
            if np.random.random() < 0.20:
                mental_health_impact.append('Depression')
            if np.random.random() < 0.15:
                mental_health_impact.append('Anxiety')
            if np.random.random() < 0.10:
                mental_health_impact.append('Loss of Appetite')
            # Suicidal thoughts (1 in 200 for teens with depression)
            if (age_group == '13-17' and
                    'Depression' in mental_health_impact and
                    np.random.random() < 0.005):
                mental_health_impact.append('Suicidal Thoughts')

        # School education received (53.4% haven't received education)
        received_education = np.random.random() < 0.466

        # Awareness level
        if received_education:
            awareness = np.random.choice(['High', 'Medium'])
        else:
            awareness = 'Medium' if np.random.random() < 0.3 else 'Low'

        # Impact certainty (50% unsure how it affected them)
        if experienced_cyberbullying:
            impact_certainty = 'Unsure' if np.random.random() < 0.50 else 'Certain'
        else:
            impact_certainty = 'Not Applicable'

        # Create student record
        student = {
            'id': f'S{i + 1:04d}',
            'type': 'Student',
            'age_group': age_group,
            'gender': gender,
            'daily_internet_hours': daily_hours,
            'primary_activity': primary_activity,
            'uses_facebook': uses_facebook,
            'num_social_media_accounts': num_accounts,
            'exposed_to_bad_language': exposed_bad_language,
            'learned_bad_words': learned_bad_words,
            'experienced_cyberbullying': experienced_cyberbullying,
            'cyberbullying_types': '; '.join(cyberbullying_types) if cyberbullying_types else 'None',
            'reported_incident': reported_incident if reported_incident is not None else 'N/A',
            'reported_to': reported_to if reported_to else 'None',
            'mental_health_impact': '; '.join(mental_health_impact) if mental_health_impact else 'None',
            'received_school_education': received_education,
            'awareness_level': awareness,
            'impact_certainty': impact_certainty
        }

        students.append(student)

    return pd.DataFrame(students)


def generate_parent_data(n_parents):
    """Generate synthetic parent data based on survey statistics"""

    parents = []

    for i in range(n_parents):
        # Parent age distribution (11.3% 25-34, 57.4% 35-44, 28.8% 45-54, 2.5% 55+)
        rand = np.random.random()
        if rand < 0.113:
            parent_age = '25-34'
        elif rand < 0.687:
            parent_age = '35-44'
        elif rand < 0.975:
            parent_age = '45-54'
        else:
            parent_age = '55+'

        # Control over child's internet use (51.8% control, 39% want to but can't, 9.2% don't)
        control_rand = np.random.random()
        if control_rand < 0.518:
            internet_control = 'Yes'
        elif control_rand < 0.908:
            internet_control = 'Want to but struggle'
        else:
            internet_control = 'No'

        # Knowledge of Facebook age requirement (37.2% don't know)
        knows_fb_age = np.random.random() < 0.628

        # Awareness of child being cyberbullied (61.3% no, 34.5% unsure, 4.2% yes)
        aware_rand = np.random.random()
        if aware_rand < 0.613:
            aware_cyberbullying = 'No'
        elif aware_rand < 0.958:
            aware_cyberbullying = 'Unsure'
        else:
            aware_cyberbullying = 'Yes'

        # Concerns about electronic device use
        concerns = []
        if np.random.random() < 0.318:
            concerns.append('Insomnia')
        if np.random.random() < 0.295:
            concerns.append('Academic Performance')
        if np.random.random() < 0.20:
            concerns.append('Depression')
        if np.random.random() < 0.15:
            concerns.append('Eating Disorders')
        if np.random.random() < 0.10:
            concerns.append('Aggression')

        # Digital literacy level
        if knows_fb_age and internet_control == 'Yes':
            digital_literacy = 'Medium' if np.random.random() < 0.6 else 'High'
        else:
            digital_literacy = 'Low' if np.random.random() < 0.7 else 'Medium'

        # Communication with child
        communicates_safety = np.random.random() < 0.55

        # Child's age group
        child_age_group = '9-12' if np.random.random() < 0.324 else '13-17'

        parent = {
            'id': f'P{i + 1:04d}',
            'type': 'Parent',
            'age_group': parent_age,
            'internet_control': internet_control,
            'knows_facebook_age_req': knows_fb_age,
            'aware_of_child_cyberbullying': aware_cyberbullying,
            'concerns_about_device_use': '; '.join(concerns) if concerns else 'None',
            'digital_literacy': digital_literacy,
            'communicates_about_safety': communicates_safety,
            'child_age_group': child_age_group
        }

        parents.append(parent)

    return pd.DataFrame(parents)


def main():
    """Main function to generate and save datasets"""

    print("=" * 60)
    print("Cyberbullying Synthetic Dataset Generator")
    print("Based on UNDP Mongolia Youth Survey (March 2025)")
    print("=" * 60)
    print()

    # Get user input
    try:
        n_records = int(input("Enter total number of records to generate (default: 1000): ") or "1000")
        include_parents = input("Include parent data? (yes/no, default: yes): ").lower() or "yes"
        include_parents = include_parents.startswith('y')
    except ValueError:
        print("Invalid input. Using defaults: 1000 records with parents.")
        n_records = 1000
        include_parents = True

    # Calculate split
    if include_parents:
        n_students = int(n_records * 0.7)
        n_parents = n_records - n_students
    else:
        n_students = n_records
        n_parents = 0

    print(f"\nGenerating {n_students} student records...")
    students_df = generate_student_data(n_students)

    if include_parents:
        print(f"Generating {n_parents} parent records...")
        parents_df = generate_parent_data(n_parents)
    else:
        parents_df = pd.DataFrame()

    # Save files
    print("\nSaving datasets...")
    students_df.to_csv('students_data.csv', index=False)
    print(f"✓ Saved: students_data.csv ({len(students_df)} records)")

    if not parents_df.empty:
        parents_df.to_csv('parents_data.csv', index=False)
        print(f"✓ Saved: parents_data.csv ({len(parents_df)} records)")

        # Combine all data
        combined_df = pd.concat([students_df, parents_df], ignore_index=True)
        combined_df.to_csv('combined_data.csv', index=False)
        print(f"✓ Saved: combined_data.csv ({len(combined_df)} records)")

    # Display statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nStudent Records: {len(students_df)}")
    print(f"  - Age 9-12: {len(students_df[students_df['age_group'] == '9-12'])}")
    print(f"  - Age 13-17: {len(students_df[students_df['age_group'] == '13-17'])}")
    print(f"  - Female: {len(students_df[students_df['gender'] == 'Female'])}")
    print(f"  - Male: {len(students_df[students_df['gender'] == 'Male'])}")

    cyberbullied = students_df[students_df['experienced_cyberbullying'] == True]
    print(f"\nCyberbullying Cases: {len(cyberbullied)} ({len(cyberbullied) / len(students_df) * 100:.1f}%)")

    reported = cyberbullied[cyberbullied['reported_incident'] == True]
    print(f"  - Reported: {len(reported)} ({len(reported) / len(cyberbullied) * 100:.1f}% of victims)")

    if not parents_df.empty:
        print(f"\nParent Records: {len(parents_df)}")
        print(f"  - Control internet use: {len(parents_df[parents_df['internet_control'] == 'Yes'])}")
        print(f"  - Struggle to control: {len(parents_df[parents_df['internet_control'] == 'Want to but struggle'])}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    # Display sample
    print("\nSample of student data (first 5 rows):")
    print(students_df.head().to_string())


if __name__ == "__main__":
    main()
