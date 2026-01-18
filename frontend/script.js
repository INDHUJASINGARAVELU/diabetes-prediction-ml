function predict() {

    // Collect values
    const age = Number(document.getElementById("age").value);
    const gender = Number(document.getElementById("gender").value);
    const pulse = Number(document.getElementById("pulse_rate").value);
    const systolic = Number(document.getElementById("systolic_bp").value);
    const diastolic = Number(document.getElementById("diastolic_bp").value);
    const glucose = Number(document.getElementById("glucose").value);
    const height = Number(document.getElementById("height").value);
    const weight = Number(document.getElementById("weight").value);
    const bmi = Number(document.getElementById("bmi").value);

    // Derived feature
    const hypertensive = (systolic >= 140 || diastolic >= 90) ? 1 : 0;

    // Payload to backend
    const data = {
        age: age,
        gender: gender,
        pulse_rate: pulse,
        systolic_bp: systolic,
        diastolic_bp: diastolic,
        glucose: glucose,
        height: height,
        weight: weight,
        bmi: bmi,
        family_diabetes: 1,
        hypertensive: hypertensive,
        family_hypertension: hypertensive,
        cardiovascular_disease: 0,
        stroke: 0
    };

    // Call backend API
    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {

        const prob = result.risk_probability;
        const percent = Math.round(prob * 100);

        // ----------------------------
        // HYBRID RISK DECISION LOGIC
        // ----------------------------
        let color = "green";
        let level = "Low Risk 🟢";
        let warning = "You are currently at low risk.";

        // 🔴 MEDICAL OVERRIDE (CRITICAL FIX)
        if (glucose >= 200 || bmi >= 30 || systolic >= 160) {
            color = "red";
            level = "High Risk 🔴";
            warning = "⚠️ High clinical indicators detected. Please consult a doctor immediately.";
        }
        // 🟠 ML-BASED MEDIUM RISK
        else if (prob >= 0.35) {
            color = "orange";
            level = "Medium Risk 🟠";
            warning = "⚠️ Moderate risk detected. Monitor your health regularly.";
        }

       
        document.getElementById("result").innerHTML =
            `Prediction: <b>${result.diabetes}</b><br>
             Risk Probability: <b>${percent}%</b>`;

        const bar = document.getElementById("progress-bar");
        bar.style.width = percent + "%";
        bar.style.background = color;

        document.getElementById("risk-level").innerText = level;
        document.getElementById("risk-level").style.color = color;

        document.getElementById("doctor-warning").innerText = warning;
        document.getElementById("doctor-warning").style.color = color;
    })
    .catch(error => {
        console.error(error);
        document.getElementById("result").innerText =
            "❌ Backend not reachable. Please start the server.";
    });
}
