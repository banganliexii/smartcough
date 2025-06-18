label_feedback = {
    'dry_cough': {
        "penjelasan": "Batuk kering biasanya muncul karena tenggorokan lagi sensitif atau infeksi ringan.",
        "saran": "Coba sering-sering minum air hangat, hindari udara dingin, dan jangan lupa istirahat cukup, ya."
    },
    'wet_cough': {
        "penjelasan": "Batuk berdahak bisa jadi tanda tubuh lagi berusaha ngeluarin lendir dari saluran napas bawah.",
        "saran": "Bisa dibantu dengan obat pengencer dahak, dan kalau belum membaik, mending periksa ke dokter, deh."
    },
    'allergy_cough': {
        "penjelasan": "Kalau batuk karena alergi, biasanya dipicu debu atau serbuk sari yang beterbangan.",
        "saran": "Usahakan hindari pemicunya, dan boleh minum antihistamin kalau perlu."
    },
    'covid_cough': {
        "penjelasan": "Batuk karena COVID biasanya barengan sama gejala lain kayak demam, badan lemas, atau kehilangan penciuman.",
        "saran": "Sebaiknya langsung cek COVID dan istirahat di rumah dulu sambil pantau kondisi."
    },
    'sneeze_flu': {
        "penjelasan": "Bersin dan flu ringan sering kejadian pas imun tubuh lagi menurun.",
        "saran": "Perbanyak waktu santai, tidur cukup, dan jangan lupa asupan vitamin C."
    }
}

def generate_feedback(label, umur='dewasa'):
    info = label_feedback.get(label, {})
    if umur == "anak":
        return f"🧒 Wah, kayaknya si virus kecil lagi ganggu, ya! Yuk bantu tubuhmu dengan madu hangat dan tidur yang nyenyak!"
    elif umur == "lansia":
        return f"🧓 {info.get('penjelasan')}\n🩺 {info.get('saran')}\nJaga kesehatan terus, ya Bapak/Ibu. Jangan terlalu capek."
    else:
        return f"🩺 {info.get('penjelasan')}\n💡 {info.get('saran')}"
