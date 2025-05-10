const N_Elec_excitation=9
const E_exc_threshold_eV = (11.184, 12.327, 8.9, 11.75, 11.8, 13.9, 13.9, 14.7, 14.9) # excitation threshold energy in eV unit

const XT_mom = Xsec_Table{100}(
   # recommended Momentum transfer Xsec for (0.001~100) eV range (by Elford)
   0.050505050505051, # dlog10E( log10(100)-log10(0.001) )/(100-1)

   # E_eV
   SVector(0.00100000000000000, 0.00112332403297803, 0.00126185688306602, 0.00141747416292680, 0.00159228279334109, 0.00178864952905744, 0.00200923300256505, 0.00225701971963392, 0.00253536449397011, 0.00284803586843580,
      0.00319926713779738, 0.00359381366380463, 0.00403701725859655, 0.00453487850812858, 0.00509413801481638, 0.00572236765935022, 0.00642807311728432, 0.00722080901838546, 0.00811130830789687, 0.00911162756115489,
      0.0102353102189903, 0.0114975699539774, 0.0129154966501488, 0.0145082877849594, 0.0162975083462064, 0.0183073828029537, 0.0205651230834865, 0.0231012970008316, 0.0259502421139974, 0.0291505306282518,
      0.0327454916287773, 0.0367837977182863, 0.0413201240011534, 0.0464158883361278, 0.0521400828799968, 0.0585702081805667, 0.0657933224657568, 0.0739072203352578, 0.0830217568131974, 0.0932603346883220,
      0.104761575278967, 0.117681195243500, 0.132194114846603, 0.148496826225447, 0.166810053720006, 0.187381742286038, 0.210490414451202, 0.236448941264541, 0.265608778294669, 0.298364724028334,
      0.335160265093884, 0.376493580679247, 0.422924287438950, 0.475081016210280, 0.533669923120631, 0.599484250318941, 0.673415065775082, 0.756463327554629, 0.849753435908645, 0.954548456661834,
      1.07226722201032, 1.20450354025878, 1.35304777457981, 1.51991108295293, 1.70735264747069, 1.91791026167249, 2.15443469003188, 2.42012826479438, 2.71858824273294, 3.05385550883342,
      3.43046928631492, 3.85352859371053, 4.32876128108306, 4.86260158006535, 5.46227721768434, 6.13590727341317, 6.89261210434970, 7.74263682681127, 8.69749002617784, 9.77009957299225,
      10.9749876549306, 12.3284673944207, 13.8488637139387, 15.5567614393047, 17.4752840000768, 19.6304065004027, 22.0513073990305, 24.7707635599171, 27.8255940220712, 31.2571584968824,
      35.1119173421513, 39.4420605943766, 44.3062145758388, 49.7702356433211, 55.9081018251222, 62.8029144183425, 70.5480231071864, 79.2482898353917, 89.0215085445038, 100.0),

   # Xsec
   SVector(7.25000000000000e-20, 7.25616620164890e-20, 7.26000000000000e-20, 7.26000000000000e-20, 7.26307609311137e-20, 7.26962165096858e-20, 7.28036932010260e-20, 7.29028078878536e-20, 7.30353644939701e-20, 7.33480358684358e-20,
      7.35597801413392e-20, 7.36781440991414e-20, 7.38259120810176e-20, 7.41744149556900e-20, 7.45282414044449e-20, 7.47167102978051e-20, 7.50568438703706e-20, 7.55104045091927e-20, 7.59556541539484e-20, 7.64669765366929e-20,
      7.70941240875961e-20, 7.75990279815909e-20, 7.81661986600595e-20, 7.88033151139838e-20, 7.96055038948963e-20, 8.05536914014768e-20, 8.16147467717249e-20, 8.25784928603160e-20, 8.37371113724388e-20, 8.52092440889958e-20,
      8.66158319026476e-20, 8.81100051557659e-20, 8.97488421603921e-20, 9.14814020342834e-20, 9.32778223775991e-20, 9.50139562087530e-20, 9.68483306164392e-20, 9.88768050838144e-20, 1.01034568930771e-19, 1.03217273631431e-19,
      1.05628507179398e-19, 1.08147833072482e-19, 1.10591705424945e-19, 1.13254481616823e-19, 1.15909441033201e-19, 1.18685809074325e-19, 1.21291003102925e-19, 1.23990689891512e-19, 1.26836007603110e-19, 1.29849554610607e-19,
      1.32847981472605e-19, 1.36195980035019e-19, 1.39727624408165e-19, 1.43430752150930e-19, 1.47354875079720e-19, 1.51566992020412e-19, 1.55270753288754e-19, 1.59479629705286e-19, 1.63739890872254e-19, 1.68145583636459e-19,
      1.72593219159356e-19, 1.77067553103882e-19, 1.79295716618697e-19, 1.81546459193557e-19, 1.81983822844098e-19, 1.81551493560801e-19, 1.78907027401547e-19, 1.75134178639920e-19, 1.69103623362782e-19, 1.61410527872098e-19,
      1.51693892413075e-19, 1.40778962282268e-19, 1.30063136969147e-19, 1.18799106660621e-19, 1.08549792238819e-19, 9.80837074448743e-20, 8.74141693286693e-20, 7.76567312223949e-20, 6.85225897643995e-20, 5.96392034160620e-20,
      5.10530854278805e-20, 4.27326268361815e-20, 3.70514125889156e-20, 3.13655198875956e-20, 2.65947937864756e-20, 2.22329878117449e-20, 1.87584271669147e-20, 1.50708446127524e-20, 1.25051759703871e-20, 1.02155931028749e-20,
      8.51564445211127e-21, 6.60605127787994e-21, 5.41693900789130e-21, 4.22031839411268e-21, 3.54374120653704e-21, 2.91940181955271e-21, 2.40643500639098e-21, 2.03232353707816e-21, 1.72033323512038e-21, 1.49000000000000e-21))


function Xsec_Electron_Momentum_Transfer!(in_Energy_eV::AbstractVector{Float64}, out_Xsec::AbstractVector{Float64}; XT=XT_mom::Xsec_Table)
    # recommended Momentum transfer Xsec for (0.001~100) eV range (by Elford)
    # for higher range (>100 eV) use Wingerden data of elastic Xsec.. and will be used with screened coulomb model

    log10_E0 = log10(XT.E_eV[1])
    inv_dlog10E = 1.0./XT.dlog10E

    for (k,E_eV) in pairs(in_Energy_eV)
       if E_eV <= 0.001
          out_Xsec[k] = XT.Xsec[1]
       elseif E_eV > 0.001 && E_eV < 100.0
          Eid = floor(Int, (log10(E_eV) - log10_E0)*inv_dlog10E) + 1
          w = (E_eV - XT.E_eV[Eid]) / (XT.E_eV[Eid+1] - XT.E_eV[Eid])
          out_Xsec[k] = (1.0 - w) * XT.Xsec[Eid] + w * XT.Xsec[Eid+1]
       elseif E_eV >= 100.0
          out_Xsec[k] = (1.2402E-18) * E_eV^(-1.10575) # Note
        #   out_Xsec[k] = 0.2 * out_Xsec[k] # to match Xsec at 100 eV
       else
          out_Xsec[k] = 0.0
       end
    end
 end

function Xsec_Electron_Momentum_Transfer(in_Energy_eV::AbstractVector{Float64}; XT=XT_mom::Xsec_Table)
   # recommended Momentum transfer Xsec for (0.001~100) eV range (by Elford)
   # for higher range (>100 eV) use Wingerden data of elastic Xsec.. and will be used with screened coulomb model
   out_Xsec = zeros(Float64, length(in_Energy_eV))

   log10_E0 = log10(XT.E_eV[1])
   inv_dlog10E = 1.0./XT.dlog10E

   for (k,E_eV) in pairs(in_Energy_eV)
      if E_eV <= 0.001
         out_Xsec[k] = XT.Xsec[1]
      elseif E_eV > 0.001 && E_eV < 100.0
         Eid = floor(Int, (log10(E_eV) - log10_E0)*inv_dlog10E) + 1
         w = (E_eV - XT.E_eV[Eid]) / (XT.E_eV[Eid+1] - XT.E_eV[Eid])
         out_Xsec[k] = (1.0 - w) * XT.Xsec[Eid] + w * XT.Xsec[Eid+1]
      elseif E_eV >= 100.0
         out_Xsec[k] = (1.2402E-18) * E_eV^(-1.10575) # Note
        #  out_Xsec[k] = 0.2 * out_Xsec[k] # to match Xsec at 100 eV
      else
         out_Xsec[k] = 0.0
      end
   end
   return out_Xsec
end


function Xsec_Electron_Momentum_Transfer(in_E_eV::Float64; XT=XT_mom::Xsec_Table)

   if in_E_eV <= 0.001
      out_Xsec = XT.Xsec[1]
   elseif in_E_eV > 0.001 && in_E_eV < 100.0
      Eid = floor(Int, (log10(in_E_eV) - log10(XT.E_eV[1])) / XT.dlog10E) + 1
      w = (in_E_eV - XT.E_eV[Eid]) / (XT.E_eV[Eid+1] - XT.E_eV[Eid])
      out_Xsec = (1.0 - w) * XT.Xsec[Eid] + w * XT.Xsec[Eid+1]
   elseif in_E_eV >= 100.0
      out_Xsec = (1.2402E-18) * in_E_eV^(-1.10575) # Note
    #   out_Xsec = 0.2 * out_Xsec # to match Xsec at 100 eV
   else
      out_Xsec = 0.0
   end

   return out_Xsec
end


function Xsec_Electron_Momentum_Transfer_vectorized!(E_eV::Vector{Float64}, out_Xsec::Vector{Float64}; XT=XT_mom::Xsec_Table)

   itp1 = LinearInterpolation(XT.E_eV, XT.Xsec)

   idx = (E_eV .<= 0.001)
   out_Xsec[idx] .= XT.Xsec[1]
   idx = (E_eV .> 0.001) .& (E_eV .< 100.0)
   out_Xsec[idx] .= itp1(E_eV[idx])
   idx = (E_eV .>= 100)
   out_Xsec[idx] .= (1.2402E-18) .* E_eV[idx] .^ (-1.10575) # Note
#    out_Xsec[idx] .= 0.2 .* out_Xsec[idx] # to match Xsec at 100 eV
end

function Xsec_Electron_Momentum_Transfer_vectorized(E_eV::Vector{Float64}; XT=XT_mom::Xsec_Table)

   itp1 = LinearInterpolation(XT.E_eV, XT.Xsec)

   out_Xsec = zeros(Float64, size(E_eV))
   idx = (E_eV .<= 0.001)
   out_Xsec[idx] .= XT.Xsec[1]
   idx = (E_eV .> 0.001) .& (E_eV .< 100.0)
   out_Xsec[idx] .= itp1(E_eV[idx])
   idx = (E_eV .>= 100)
   out_Xsec[idx] .= (1.2402E-18) .* E_eV[idx] .^ (-1.10575) # Note
#    out_Xsec[idx] .= 0.2 .* out_Xsec[idx] # to match Xsec at 100 eV

   return out_Xsec
end


function Xsec_Electron_Ionization(in_Energy_eV::Vector{Float64})
   out_Xsec = zeros(Float64, length(in_Energy_eV))
   Xsec_Electron_Ionization!(in_Energy_eV, out_Xsec)
   return out_Xsec
end

function Xsec_Electron_Ionization!(in_Energy_eV::AbstractVector{Float64}, out_Xsec::AbstractVector{Float64})
    @assert length(in_Energy_eV) == length(out_Xsec) "Length of input and output arrays should be same"

    for (k,E_eV) in pairs(in_Energy_eV)
       if E_eV < 15.46
          out_Xsec[k] =  0.0;
       elseif E_eV < 16.0
          out_Xsec[k] =  ((16.69E-23) * (E_eV - 15.43)) / (0.53.^ 2);
             elseif E_eV < 70.0
          out_Xsec[k] =  ((-4.0855E-21) + (3.3057E-22) * (E_eV)+(-3.852E-24) * (E_eV.^ 2) +
             (-5.1458E-26) * (E_eV.^ 3) + (1.4436E-27) * (E_eV.^ 4) +
             (-8.5308E-30) * (E_eV.^ 5)) / (0.53.^ 2);
       elseif E_eV < 1000.0
          out_Xsec[k] =  ((1.7117E-22) + (8.398E-22) * exp(-(E_eV + 61.36476) / 226.35342) +
             (2.2991E-21) * exp(-(E_eV + 61.36476) / 226.36454) +
             (1.111E-21) * exp(-(E_eV + 61.36476) / 1289.79864)) / (0.53.^ 2);
       elseif E_eV < 100000.
          out_Xsec[k] =   10.0.^(-0.8442*log10(E_eV) - 18.0784);
       else
          out_Xsec[k] =  0.0;
       end
    end
 end

function Xsec_Electron_Excitation(in_Energy_eV::Vector{Float64}, reaction_flag::Int)
    # Excitations
    # 1: X -> B1sigma || 2: X -> C1pi      || 3: X -> b3sigma   || 4: X -> c3pi
    # 5: X -> a3sigma || 6: X -> B'1sigma  || 7: X -> D1pi      || 8: X -> B''1sigma
    # 9: X -> D'1pi

    out_Xsec = zeros(Float64, length(in_Energy_eV))
    Xsec_Electron_Excitation!(in_Energy_eV, reaction_flag, out_Xsec)

    return out_Xsec
 end

function Xsec_Electron_Excitation!(in_Energy_eV::AbstractVector{Float64}, reaction_flag::Int, out_Xsec::AbstractVector{Float64})
   # Excitations
   # 1: X -> B1sigma || 2: X -> C1pi      || 3: X -> b3sigma   || 4: X -> c3pi
   # 5: X -> a3sigma || 6: X -> B'1sigma  || 7: X -> D1pi      || 8: X -> B''1sigma
   # 9: X -> D'1pi
   @assert length(in_Energy_eV) == length(out_Xsec) "Length of input and output arrays should be same"

   if reaction_flag==1
      # 1: X -> B1sigma
      # E_threshold = 11.184
      #!-----------------------------------------------
      #! electron-D2 X -> B1sig electronic excitation
      #! cross-sections from R. Celiberto et al.,
      #! Atomic Data and Nuclear Data Tables 77, 161-213(2001)
      #!-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)

          if E_eV < 12.0
              out_Xsec[k] =  0.0;

          elseif E_eV <= 19.0
              out_Xsec[k] =   (0.42566 * (E_eV-11.779) * exp(-0.77443*(E_eV-11.779)) + 0.12348*((E_eV-11.779).^0.52783))* (1.0E-20);

          elseif E_eV <= 20.0
              out_Xsec[k] =   ( (0.3724-0.362) * (E_eV-19) + 0.362 )*(1.0E-20);

          elseif E_eV > 20.0
              out_Xsec[k] =   ( (1.3204/(E_eV/11.184)) * ((((E_eV/11.184)-1)/((E_eV/11.184)+1)).^0.65554) * ( -0.41817 + 2.4241 * (1-1/2/(E_eV/11.184)) * log((1.5734+((E_eV/11.184)-1).^0.5)) ) )*(1.0E-20);

          end
      end

   elseif reaction_flag==2
      #  2: X -> C1pi
      #E_threshold = 12.327
      #-----------------------------------------------
      # X -> C1pi electronic excitation
      # cross-sections from R. Celiberto et al.,
      #Atomic Data and Nuclear Data Tables 77, 161-213(2001)
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 14.0 )
              out_Xsec[k] =  0.0;
          elseif (E_eV <= 20.0 )
              out_Xsec[k] =  1e-20*(-21.104*(E_eV-12.982)*exp(-7.5183*(E_eV-12.982))+0.33273*((E_eV-12.982).^(-0.12174))+
                  0.15650*(E_eV-12.982)*exp(-0.82171*(E_eV-12.982)) )
          elseif (E_eV <= 21.0)
              out_Xsec[k] =  ( (0.3005-0.2659) * (E_eV-20) + 0.2659 ) * (1.0E-20);
          elseif (E_eV > 21.0)
              out_Xsec[k] = 1e-20*( (1.2634/(((E_eV/12.286).^0.97791))) * ((((E_eV/12.286)-1)/((E_eV/12.286)+1)).^0.84579) *
                  ( 0.53604 + 1.7887 * (1-1/2/(E_eV/12.286)) * log(0.9135+(((E_eV/12.286)-1).^0.5)) ) )
          else
              out_Xsec[k] =  0.0;
          end
      end

   elseif reaction_flag==3
      #  3: X -> b3sigma
      #E_threshold = 8.9
      #!-----------------------------------------------
      #! X -> b3sig electronic excitation
      #! cross-sections from Bolsig input data
      #!-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 8.9 )
              out_Xsec[k] =   0.0;
          elseif (E_eV <= 12.0 )
              out_Xsec[k] =   ( -1.29+0.18103*E_eV-0.00405*(E_eV.^2)  ) * (1.0E-20);
          elseif (E_eV <= 17.0)
              out_Xsec[k] =   ( 0.2162+0.0069*E_eV ) * (1.0E-20);
          elseif (E_eV <= 100.0)
              out_Xsec[k] =   ( (0.13518/(0.13518-0.09322)) * (exp(-0.09322*E_eV)-exp(-0.13518*E_eV)) ) * (1.0E-20);
          elseif (E_eV > 100.0)
              out_Xsec[k] =   0.0;
          end
      end

   elseif reaction_flag==4
      # 4: X -> c3pi
      #E_threshold = 11.75
      #-----------------------------------------------
      # X -> c3pi electronic excitation
      # cross-sections from Bolsig input data
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 11.75)
              out_Xsec[k] =   0.0;
          elseif (E_eV <= 13.5)
              out_Xsec[k] =   (0.85359*(E_eV-11.74)*exp(-1.99352*(E_eV-11.74))+0.13358*((E_eV-11.74).^(1.18804))-
                  0.18351*(E_eV-11.74)*exp(-0.37682*(E_eV-11.74)) ) * (1.0E-20);
          else
              out_Xsec[k] =   0.0;
          end
      end

   elseif reaction_flag==5
      #5: X -> a3sigma
      #E_threshold = 11.8
      #-----------------------------------------------
      # X -> a3sig electronic excitation
      # cross-sections from Bolsig input data
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 11.8)
              out_Xsec[k] =   0.0;
          elseif ( E_eV <= 13.0)
              out_Xsec[k] =   ( -1.08167 + 0.09167*E_eV)* (1.0E-20);
          elseif ( E_eV <= 17.0)
              out_Xsec[k] =   ( -13.943 + 3.64767*E_eV - 0.35375*(E_eV.^2) + 0.01533*(E_eV.^3) - 0.00025*(E_eV.^4) ) * (1.0E-20);
          elseif ( E_eV <= 70.0)
              out_Xsec[k] =   ( -0.00447 + 0.15415*exp(-(E_eV-10.84186)/6.67452) + 0.07957*exp(-(E_eV-10.84186)/20.58985) ) * (1.0E-20);
          else
              out_Xsec[k] =   0.0;
          end
      end

   elseif reaction_flag==6
      #  6: X -> B'1sigma
      #E_threshold = 13.9
      #-----------------------------------------------
      # X -> B'1sig electronic excitation
      # cross-sections from R. Celiberto et al.,
      # Atomic Data and Nuclear Data Tables 77, 161-213(2001)
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 13.9 )
              out_Xsec[k] =   0.0;
          elseif (E_eV <= 22.0 )
              out_Xsec[k] =   (0.23216*(E_eV-13.87642)*exp(-0.58222*(E_eV-13.87642))+0.0164*((E_eV-13.87642).^(0.59414))- 0.17416*(E_eV-13.87642)*exp(-0.55784*(E_eV-13.87642)) ) * (1.0E-20);
          elseif (E_eV <= 200.0 )
              out_Xsec[k] =  1e-20* ( (1.97053/(((E_eV/7.33785).^0.65748))) * ((((E_eV/7.33785)-1)/((E_eV/7.33785)+1)).^3.35052 )
                  *( 2.77512 - 0.31478 * (1-1/2/(E_eV/7.33785)) * log(3775.72+(((E_eV/7.33785)-1).^0.5)) ) )
          else
              out_Xsec[k] =   0.0;
          end
      end

   elseif reaction_flag==7
      #   7: X -> D1pi
      #E_threshold = 13.9
      #-----------------------------------------------
      # X -> D1pi electronic excitation
      # cross-sections from R. Celiberto et al.,
      # Atomic Data and Nuclear Data Tables 77, 161-213(2001)
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 13.9 )
              out_Xsec[k] =   0.0;
          elseif (E_eV <= 25.0 )
              out_Xsec[k] =   1e-20*(0.15964*(E_eV-13.89051)*exp(-0.92561*(E_eV-13.89051))+0.02979*((E_eV-13.89051).^(0.33597))-
                  0.22154*(E_eV-13.89051)*exp(-1.53561*(E_eV-13.89051)) )
          elseif (E_eV <= 200.0 )
              out_Xsec[k] =   1e-20*( (3.32695/(((E_eV/5.9768).^0.69512))) * ((((E_eV/5.9768)-1)/((E_eV/5.9768)+1)).^4.69437) *
                  ( 2.93945 - 0.31565 * (1-1/2/(E_eV/5.9768)) * log(5687.78448+(((E_eV/5.9768)-1).^0.5)) ) )
          else
              out_Xsec[k] =   0.0;
          end
      end

   elseif reaction_flag==8
      #  8: X -> B''1sigma
      #E_threshold = 14.7
      #-----------------------------------------------
      # X -> B"1sig electronic excitation
      # cross-sections from R. Celiberto et al.,
      # Atomic Data and Nuclear Data Tables 77, 161-213(2001)
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 14.7 )
              out_Xsec[k] =   0.0;
          elseif (E_eV <= 22.0 )
              out_Xsec[k] =  1e-20*(0.18354*(E_eV-14.65758)*exp(-0.53837*(E_eV-14.65758))+0.00708*((E_eV-14.65758).^(0.583))-
                  0.16135*(E_eV-14.65758)*exp(-0.52336*(E_eV-14.65758)) )
          elseif (E_eV <= 200.0 )
              out_Xsec[k] =  1e-20* ( (0.82715/(((E_eV/7.29033).^0.59361))) * ((((E_eV/7.29033)-1)/((E_eV/7.29033)+1)).^3.47936) *
                  ( 2.87251 - 0.31094 * (1-1/2/(E_eV/7.29033)) * log(6794.31315+(((E_eV/7.29033)-1).^0.5)) ) )
          else
              out_Xsec[k] =   0.0;
          end
      end

   elseif reaction_flag==9
      #  9: X -> D'1pi
      #E_threshold =14.9
      #-----------------------------------------------
      # X -> D'1pi electronic excitation
      # cross-sections from R. Celiberto et al.,
      # Atomic Data and Nuclear Data Tables 77, 161-213(2001)
      #-----------------------------------------------
      for (k,E_eV) in pairs(in_Energy_eV)
          if (E_eV < 14.9 )
              out_Xsec[k] =   0.0;
          elseif (E_eV <= 27.0 )
              out_Xsec[k] =   1e-20*(0.06117*(E_eV-14.80461)*exp(-0.78927*(E_eV-14.80461))+0.01027*((E_eV-14.80461).^(0.38943))-
                  0.03799*(E_eV-14.80461)*exp(-0.94223*(E_eV-14.80461)) )
          elseif (E_eV <= 200.0 )
              out_Xsec[k] =   1e-20*( (2.27572/(((E_eV/4.76266).^0.65726))) * ((((E_eV/4.76266)-1)/((E_eV/4.76266)+1)).^6.4372) *
                  ( 2.88752 - 0.31198 * (1-1/2/(E_eV/4.76266)) * log(6950.34662+(((E_eV/4.76266)-1).^0.5)) ) )
          else
              out_Xsec[k] =   0.0;
          end
      end
   end
end

function Xsec_Electron_tot_Excitation(in_Energy_eV::Vector{Float64})
    out_Xsec = zeros(Float64, length(in_Energy_eV))
    tmp_Xsec = similar(out_Xsec)

    for i in 1:N_Elec_excitation
       Xsec_Electron_Excitation!(in_Energy_eV,i, tmp_Xsec)
       @. out_Xsec += tmp_Xsec
    end
    return out_Xsec
 end

 function Xsec_Electron_tot_Excitation!(in_Energy_eV::Vector{Float64}, out_Xsec::Vector{Float64})
    @. out_Xsec = 0.0
    tmp_Xsec = similar(out_Xsec)

    for i in 1:N_Elec_excitation
       Xsec_Electron_Excitation!(in_Energy_eV,i, tmp_Xsec)
       @. out_Xsec += tmp_Xsec
    end
 end