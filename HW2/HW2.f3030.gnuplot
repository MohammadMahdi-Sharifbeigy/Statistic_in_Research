set table "HW2.f3030.table"; set format "%.5f"
set format "%.7e";; set samples 100; set dummy x; plot [x=0.01:5] (gamma((30+30)/2.0)/(gamma(30/2.0)*gamma(30/2.0))) * (30./30.)**(30/2.0) * (x)**(30/2.0-1.0) / ((1+(30./30.)*x)**((30+30)/2.0));
