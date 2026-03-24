// Heart Disease Risk Predictor JavaScript

class HeartDiseasePredictor {
    constructor() {
        this.riskData = {
            risk_factors: {
                age_high_risk: 60,
                cholesterol_high: 240,
                bp_high: 140,
                max_heart_rate_low: 150
            },
            risk_weights: {
                age: 0.20,
                sex: 0.15,
                chest_pain: 0.25,
                blood_pressure: 0.08,
                cholesterol: 0.10,
                fasting_bs: 0.05,
                resting_ecg: 0.03,
                max_heart_rate: 0.12,
                exercise_angina: 0.15,
                st_depression: 0.10,
                st_slope: 0.08,
                vessels: 0.12,
                thalassemia: 0.07
            },
            population_stats: {
                average_age: 54,
                average_cholesterol: 246,
                average_bp: 131,
                average_heart_rate: 149
            }
        };

        this.riskGauge = null;
        this.factorsChart = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeTabs();
        this.updateSliderValues();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // Form reset
        document.getElementById('resetForm').addEventListener('click', () => {
            this.resetForm();
        });

        // Slider updates
        document.getElementById('age').addEventListener('input', (e) => {
            document.getElementById('ageValue').textContent = e.target.value;
        });

        document.getElementById('maxHeartRate').addEventListener('input', (e) => {
            document.getElementById('maxHRValue').textContent = e.target.value;
        });

        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
    }

    updateSliderValues() {
        document.getElementById('ageValue').textContent = document.getElementById('age').value;
        document.getElementById('maxHRValue').textContent = document.getElementById('maxHeartRate').value;
    }

    initializeTabs() {
        this.switchTab('parameters');
    }

    switchTab(tabName) {
        // Remove active class from all tab buttons and contents
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Add active class to selected tab button and content
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(tabName).classList.add('active');
    }

    validateForm() {
        const form = document.getElementById('predictionForm');
        const formData = new FormData(form);
        const errors = [];

        // Check required fields
        const requiredFields = [
            'sex', 'chestPain', 'bloodPressure', 'cholesterol', 
            'restingECG', 'stDepression', 'stSlope', 'vessels', 'thalassemia'
        ];

        requiredFields.forEach(field => {
            const value = formData.get(field);
            if (!value || value === '') {
                errors.push(`Please select/enter ${field.replace(/([A-Z])/g, ' $1').toLowerCase()}`);
                document.querySelector(`[name="${field}"]`).parentElement.classList.add('error');
            } else {
                document.querySelector(`[name="${field}"]`).parentElement.classList.remove('error');
            }
        });

        // Validate numeric ranges
        const bloodPressure = parseInt(formData.get('bloodPressure'));
        if (bloodPressure && (bloodPressure < 80 || bloodPressure > 220)) {
            errors.push('Blood pressure must be between 80-220 mmHg');
        }

        const cholesterol = parseInt(formData.get('cholesterol'));
        if (cholesterol && (cholesterol < 100 || cholesterol > 600)) {
            errors.push('Cholesterol must be between 100-600 mg/dl');
        }

        const stDepression = parseFloat(formData.get('stDepression'));
        if (stDepression && (stDepression < 0 || stDepression > 6)) {
            errors.push('ST Depression must be between 0-6.0');
        }

        return { isValid: errors.length === 0, errors };
    }

    handlePrediction() {
        const validation = this.validateForm();
        
        if (!validation.isValid) {
            alert('Please correct the following errors:\n' + validation.errors.join('\n'));
            return;
        }

        // Show loading state
        document.getElementById('predictionForm').classList.add('loading');

        // Simulate processing delay for better UX
        setTimeout(() => {
            const formData = this.getFormData();
            const riskAssessment = this.calculateRisk(formData);
            this.displayResults(riskAssessment, formData);
            
            // Remove loading state
            document.getElementById('predictionForm').classList.remove('loading');
        }, 1000);
    }

    getFormData() {
        const form = document.getElementById('predictionForm');
        const formData = new FormData(form);
        
        return {
            age: parseInt(formData.get('age')),
            sex: parseInt(formData.get('sex')),
            chestPain: parseInt(formData.get('chestPain')),
            bloodPressure: parseInt(formData.get('bloodPressure')),
            cholesterol: parseInt(formData.get('cholesterol')),
            fastingBS: formData.get('fastingBS') ? 1 : 0,
            restingECG: parseInt(formData.get('restingECG')),
            maxHeartRate: parseInt(formData.get('maxHeartRate')),
            exerciseAngina: formData.get('exerciseAngina') ? 1 : 0,
            stDepression: parseFloat(formData.get('stDepression')),
            stSlope: parseInt(formData.get('stSlope')),
            vessels: parseInt(formData.get('vessels')),
            thalassemia: parseInt(formData.get('thalassemia'))
        };
    }

    calculateRisk(data) {
        const weights = this.riskData.risk_weights;
        let totalRisk = 0;
        let factorContributions = [];

        // Age factor
        const ageRisk = (data.age / 100) * weights.age;
        totalRisk += ageRisk;
        factorContributions.push({ name: 'Age', value: ageRisk * 100, percentage: (ageRisk / weights.age) * 100 });

        // Sex factor (male = higher risk)
        const sexRisk = data.sex * weights.sex;
        totalRisk += sexRisk;
        factorContributions.push({ name: 'Sex', value: sexRisk * 100, percentage: (sexRisk / weights.sex) * 100 });

        // Chest pain factor (higher values = higher risk)
        const chestPainRisk = (data.chestPain / 3) * weights.chest_pain;
        totalRisk += chestPainRisk;
        factorContributions.push({ name: 'Chest Pain', value: chestPainRisk * 100, percentage: (chestPainRisk / weights.chest_pain) * 100 });

        // Blood pressure factor
        const bpRisk = Math.min(data.bloodPressure / this.riskData.risk_factors.bp_high, 1) * weights.blood_pressure;
        totalRisk += bpRisk;
        factorContributions.push({ name: 'Blood Pressure', value: bpRisk * 100, percentage: (bpRisk / weights.blood_pressure) * 100 });

        // Cholesterol factor
        const cholRisk = Math.min(data.cholesterol / this.riskData.risk_factors.cholesterol_high, 1) * weights.cholesterol;
        totalRisk += cholRisk;
        factorContributions.push({ name: 'Cholesterol', value: cholRisk * 100, percentage: (cholRisk / weights.cholesterol) * 100 });

        // Fasting blood sugar
        const fbsRisk = data.fastingBS * weights.fasting_bs;
        totalRisk += fbsRisk;
        factorContributions.push({ name: 'Fasting BS', value: fbsRisk * 100, percentage: (fbsRisk / weights.fasting_bs) * 100 });

        // Resting ECG
        const ecgRisk = (data.restingECG / 2) * weights.resting_ecg;
        totalRisk += ecgRisk;
        factorContributions.push({ name: 'Resting ECG', value: ecgRisk * 100, percentage: (ecgRisk / weights.resting_ecg) * 100 });

        // Max heart rate (lower = higher risk)
        const hrRisk = Math.max(0, (this.riskData.risk_factors.max_heart_rate_low - data.maxHeartRate) / this.riskData.risk_factors.max_heart_rate_low) * weights.max_heart_rate;
        totalRisk += hrRisk;
        factorContributions.push({ name: 'Max Heart Rate', value: hrRisk * 100, percentage: (hrRisk / weights.max_heart_rate) * 100 });

        // Exercise angina
        const exAnginaRisk = data.exerciseAngina * weights.exercise_angina;
        totalRisk += exAnginaRisk;
        factorContributions.push({ name: 'Exercise Angina', value: exAnginaRisk * 100, percentage: (exAnginaRisk / weights.exercise_angina) * 100 });

        // ST depression
        const stRisk = (data.stDepression / 6) * weights.st_depression;
        totalRisk += stRisk;
        factorContributions.push({ name: 'ST Depression', value: stRisk * 100, percentage: (stRisk / weights.st_depression) * 100 });

        // ST slope (flat and downsloping = higher risk)
        const slopeRisk = ((2 - data.stSlope) / 2) * weights.st_slope;
        totalRisk += slopeRisk;
        factorContributions.push({ name: 'ST Slope', value: slopeRisk * 100, percentage: (slopeRisk / weights.st_slope) * 100 });

        // Number of vessels
        const vesselRisk = (data.vessels / 3) * weights.vessels;
        totalRisk += vesselRisk;
        factorContributions.push({ name: 'Major Vessels', value: vesselRisk * 100, percentage: (vesselRisk / weights.vessels) * 100 });

        // Thalassemia (reversible defect = highest risk)
        const thalRisk = ((data.thalassemia - 1) / 2) * weights.thalassemia;
        totalRisk += thalRisk;
        factorContributions.push({ name: 'Thalassemia', value: thalRisk * 100, percentage: (thalRisk / weights.thalassemia) * 100 });

        // Convert to percentage and determine risk level
        const riskPercentage = Math.min(totalRisk * 100, 100);
        let riskLevel, riskClass, riskDescription;

        if (riskPercentage < 30) {
            riskLevel = 'Low Risk';
            riskClass = 'risk-low';
            riskDescription = 'Your cardiovascular risk is within normal range. Continue maintaining healthy lifestyle habits.';
        } else if (riskPercentage < 60) {
            riskLevel = 'Medium Risk';
            riskClass = 'risk-medium';
            riskDescription = 'Your cardiovascular risk is elevated. Consider lifestyle modifications and consult with a healthcare provider.';
        } else {
            riskLevel = 'High Risk';
            riskClass = 'risk-high';
            riskDescription = 'Your cardiovascular risk is significantly elevated. Please consult with a healthcare provider immediately.';
        }

        return {
            percentage: Math.round(riskPercentage),
            level: riskLevel,
            class: riskClass,
            description: riskDescription,
            factors: factorContributions.sort((a, b) => b.value - a.value)
        };
    }

    displayResults(riskAssessment, formData) {
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

        // Update risk display
        document.getElementById('riskLevel').textContent = riskAssessment.level;
        document.getElementById('riskLevel').className = `risk-level ${riskAssessment.class}`;
        
        document.getElementById('riskPercentage').textContent = `${riskAssessment.percentage}%`;
        document.getElementById('riskPercentage').className = `risk-percentage ${riskAssessment.class}`;
        
        document.getElementById('riskDescription').textContent = riskAssessment.description;

        // Create risk gauge chart
        this.createRiskGauge(riskAssessment.percentage, riskAssessment.class);

        // Create factors chart
        this.createFactorsChart(riskAssessment.factors);

        // Display recommendations
        this.displayRecommendations(riskAssessment, formData);
    }

    createRiskGauge(percentage, riskClass) {
        const canvas = document.getElementById('riskGauge');
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.riskGauge) {
            this.riskGauge.destroy();
        }

        // Determine color based on risk level
        let color;
        if (riskClass === 'risk-low') {
            color = '#059669'; // Medical green
        } else if (riskClass === 'risk-medium') {
            color = '#ea580c'; // Medical orange
        } else {
            color = '#dc2626'; // Medical red
        }

        this.riskGauge = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [percentage, 100 - percentage],
                    backgroundColor: [color, '#e5e7eb'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }

    createFactorsChart(factors) {
        const canvas = document.getElementById('factorsChart');
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.factorsChart) {
            this.factorsChart.destroy();
        }

        // Take top 8 factors for better visibility
        const topFactors = factors.slice(0, 8);

        this.factorsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: topFactors.map(f => f.name),
                datasets: [{
                    label: 'Risk Contribution (%)',
                    data: topFactors.map(f => f.percentage),
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325'],
                    borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    displayRecommendations(riskAssessment, formData) {
        const recommendations = this.generateRecommendations(riskAssessment, formData);
        const container = document.getElementById('recommendationList');
        
        container.innerHTML = recommendations.map(rec => `
            <div class="recommendation-item">
                <div class="recommendation-icon">
                    <i class="${rec.icon}"></i>
                </div>
                <div class="recommendation-text">
                    <div class="recommendation-title">${rec.title}</div>
                    <div class="recommendation-description">${rec.description}</div>
                </div>
            </div>
        `).join('');
    }

    generateRecommendations(riskAssessment, formData) {
        const recommendations = [];

        // Base recommendations for all risk levels
        recommendations.push({
            icon: 'fas fa-user-md',
            title: 'Consult Healthcare Provider',
            description: 'Discuss these results with your doctor for professional medical advice and personalized treatment plans.'
        });

        // Risk level specific recommendations
        if (riskAssessment.class === 'risk-high') {
            recommendations.push({
                icon: 'fas fa-exclamation-triangle',
                title: 'Immediate Medical Attention',
                description: 'Your risk level indicates urgent need for medical evaluation. Schedule an appointment as soon as possible.'
            });
        }

        // Specific factor-based recommendations
        if (formData.cholesterol > this.riskData.risk_factors.cholesterol_high) {
            recommendations.push({
                icon: 'fas fa-apple-alt',
                title: 'Dietary Modifications',
                description: 'Consider a heart-healthy diet low in saturated fats and cholesterol. Consult a nutritionist for personalized meal plans.'
            });
        }

        if (formData.bloodPressure > this.riskData.risk_factors.bp_high) {
            recommendations.push({
                icon: 'fas fa-tachometer-alt',
                title: 'Blood Pressure Management',
                description: 'Monitor your blood pressure regularly and discuss medication options with your healthcare provider.'
            });
        }

        if (formData.maxHeartRate < this.riskData.risk_factors.max_heart_rate_low) {
            recommendations.push({
                icon: 'fas fa-running',
                title: 'Exercise Program',
                description: 'Consider a supervised exercise program to improve cardiovascular fitness and heart rate response.'
            });
        }

        if (formData.exerciseAngina === 1) {
            recommendations.push({
                icon: 'fas fa-heartbeat',
                title: 'Exercise Monitoring',
                description: 'Exercise under medical supervision and learn to recognize symptoms of exercise-induced angina.'
            });
        }

        // Lifestyle recommendations
        recommendations.push({
            icon: 'fas fa-leaf',
            title: 'Lifestyle Changes',
            description: 'Maintain a healthy weight, quit smoking if applicable, limit alcohol consumption, and manage stress effectively.'
        });

        return recommendations;
    }

    resetForm() {
        // Reset form
        document.getElementById('predictionForm').reset();
        
        // Update slider values
        this.updateSliderValues();
        
        // Hide results
        document.getElementById('resultsSection').style.display = 'none';
        
        // Remove error classes
        document.querySelectorAll('.form-group.error').forEach(group => {
            group.classList.remove('error');
        });

        // Destroy charts
        if (this.riskGauge) {
            this.riskGauge.destroy();
            this.riskGauge = null;
        }
        
        if (this.factorsChart) {
            this.factorsChart.destroy();
            this.factorsChart = null;
        }

        // Scroll back to top
        document.querySelector('.header').scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new HeartDiseasePredictor();
});