from rag_pipeline.analytics_engine import AnalyticsEngine
import datetime

def main():
    engine = AnalyticsEngine()
    report = engine.generate_report()
    
    print("\n" + "#"*50)
    print(f"📊 RAG SYSTEM HEALTH DASHBOARD - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("#"*50 + "\n")
    
    if "message" in report:
        print(report["message"])
        return

    print(f"**Total Queries Analyzed**: {report['total_queries']}")
    print(f"**Overall Failure Rate**: {report['failure_rate'] * 100}%")
    print(f"**Avg Quality Score**: {report['avg_quality_score']}/1.0")
    
    print("\n### 📈 Metric Trends")
    for k, v in report['metric_trends'].items():
        print(f"- **{k.replace('_', ' ').title()}**: {v}")
        
    print("\n### 🚨 Top Failure Drivers")
    for k, v in report['top_failure_drivers'].items():
        print(f"- **{k}**: {v} failures")
        
    print("\n### 🐛 Recent Failure Diagnostics")
    failures_list = report.get("top_failed_queries", [])
    if failures_list:
        for f in failures_list:
            print(f"- **ID**: {f.get('id')}")
            print(f"  **Question**: {f.get('question')}")
            print(f"  **Failure**: {f.get('failure')} -> {f.get('reason')}")
            print(f"  **Suggested Fix**: {f.get('fix_suggested')}")
            print("  " + "-"*30)
    else:
        print("  ✅ No critical failures analyzed recently.")
            
    print("\n### 💰 Financial Optimization")
    fin = report.get("financial_impact", {})
    print(f"- **Wasted Spend Detected**: {fin.get('wasted_spend_detected')}")
    print(f"- **Projected Annual Waste**: {fin.get('projected_annual_waste')}")

    print("\n### 💡 Strategic Recommendations")
    for rec in report['strategic_recommendations']:
        print(f"✅ {rec}")

if __name__ == "__main__":
    main()
