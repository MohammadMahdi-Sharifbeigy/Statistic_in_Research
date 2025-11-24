set table "HW2.f102.table"; set format "%.5f"
set format "%.7e";; set samples 100; set dummy x; plot [x=0.01:8] (gamma((10+2)/2.0)/(gamma(10/2.0)*gamma(2/2.0))) * (10./2.)**(10/2.0) * (x)**(10/2.0-1.0) / ((1+(10./2.)*x)**((10+2)/2.0));
