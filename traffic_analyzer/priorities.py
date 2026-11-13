"""Human-readable review priorities with explicit reasons."""
CHECK = "Проверить"
FIRST = "Проверить в первую очередь"


def event_priority(event):
    first=event.severity in {"Высокий",FIRST}
    reason="Сработала проверка; требуется ручная оценка."
    duration=int((event.end-event.start).total_seconds()/3600)+1
    if event.category=="Трафик":
        if event.kind=="Низкая скорость при высокой загрузке":
            first=True;reason="Одновременно нарушены пороги скорости и загрузки при наличии потока."
        elif event.kind=="Аномальный суточный объём":
            first=True;reason="Отклонение полного суточного объёма прошло статистический и практический пороги."
        elif "форма" in event.kind.lower():
            first=event.value is not None and event.value<.4
            reason="Сильное искажение формы: корреляция ниже 0,40." if first else "Необычная форма дня требует проверки контекста."
        elif "распределение" in event.kind.lower():
            first=event.value is not None and event.expected is not None and abs(event.value-event.expected)>=40
            reason="Доли направлений отличаются минимум на 40 п.п." if first else "Нетипичное распределение направлений."
        elif event.scope=="Часы":
            strong_z=event.score is not None and event.score>=6
            relative=abs((event.value-event.expected)/event.expected)*100 if event.value is not None and event.expected is not None and event.expected>0 else 0
            first=duration>=2 or strong_z or relative>=50
            reasons=[]
            if duration>=2:reasons.append(f"длительность {duration} ч")
            if strong_z:reasons.append("робастный Z ≥ 6")
            if relative>=50:reasons.append("отклонение от нормы ≥ 50%")
            reason="; ".join(reasons)+"." if reasons else "Одиночный часовой сигнал умеренной силы."
    elif event.category=="Качество данных":
        if event.kind=="Неполный или непригодный месяц":
            reason="Пригодность месяца ниже установленного порога."
        elif event.kind=="Систематическое расхождение учёта":
            reason="Повторяются значительные расхождения учёта." if first else "Повторяются небольшие расхождения учёта."
        else:
            reason=f"Дефект затрагивает {duration} последовательных часов." if first else f"Локальный дефект качества ({duration} ч)."
    elif event.scope=="Период":
        first=True;reason="Устойчивое изменение на нескольких полных сутках."
    return (FIRST if first else CHECK),reason


def promote_correlated_events(frame):
    """Confirm traffic events using overlapping volume and speed evidence only."""
    result=frame.copy()
    traffic=result[result["Категория"].eq("Трафик") & result["Масштаб"].eq("Часы")]
    for (_, _),group in traffic.groupby(["Участок дороги","Дата"]):
        volume=group[group["_metric"].str.startswith("vol_")]
        speed=group[group["_metric"].str.startswith("speed_")]
        for vi,v in volume.iterrows():
            for si,s in speed.iterrows():
                compatible=v["Направление"] in {"Оба",s["Направление"]}
                if compatible and v["Начало"]<=s["Конец"] and s["Начало"]<=v["Конец"]:
                    for index in (vi,si):
                        if result.at[index,"Уровень"]!=FIRST:
                            result.at[index,"Уровень"]=FIRST
                            result.at[index,"Пояснение"]+=" Подтверждение: во времени совпали независимые сигналы интенсивности и скорости."
    return result
