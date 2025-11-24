set table "HW2.f55.table"; set format "%.5f"
set format "%.7e";; set samples 100; set dummy x; plot [x=0.01:5] (gamma((5+5)/2.0)/(gamma(5/2.0)*gamma(5/2.0))) * (5./5.)**(5./2.0) * (x)**(5./2.0-1.0) / ((1+(5./5.)*x)**((5+5)/2.0));
